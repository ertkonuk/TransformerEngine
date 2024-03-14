/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "userbuffers.h"
#include <assert.h>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <immintrin.h>
#include <iostream>
#include <math.h>
#include <sched.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <x86intrin.h>

#ifdef NCCLBOOTSTRAP
inline void ub_barrier(ncclComm_t c) {
  void* dummy;
  cudaMalloc(&dummy,4);
  ncclAllReduce(dummy,dummy,1, ncclInt, ncclSum, c);
  cudaDeviceSynchronize();
  cudaFree(dummy);
}
#else
inline void ub_barrier(MPI_Comm c) {
  MPI_Barrier(c);
}
#endif


#ifndef NCCLBOOTSTRAP
static int oob_bcast(void *comm_context, void *buf, int size, int root) {
  MPI_Bcast(buf, size, MPI_BYTE, root,
            (reinterpret_cast<communicator *>(comm_context))->comm_inter);
  return 0;
}
static int oob_gather(void *comm_context, int root, void *sbuf, void *rbuf, int len) {
  MPI_Gather(sbuf, len, MPI_BYTE, rbuf, len, MPI_BYTE, root,
             (reinterpret_cast<communicator *>(comm_context))->comm_inter);
  return 0;
}
#endif

static int oob_barrier(void *comm_context) {
  ub_barrier((reinterpret_cast<communicator *>(comm_context))->comm_inter);
  return 0;
}
int stringCmp(const void *a, const void *b) { return strcmp((const char *)a, (const char *)b); }

#define CUDACHECK(cmd)                                                                             \
  do {                                                                                             \
    cudaError_t e = cmd;                                                                           \
    if (e != cudaSuccess) {                                                                        \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));        \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  } while (0)

#define CUCHECK(cmd)                                                                               \
  do {                                                                                             \
    CUresult retval = cmd;                                                                         \
    if (retval != CUDA_SUCCESS) {                                                                  \
      const char *error_string;                                                                    \
      cuGetErrorString(retval, &error_string);                                                     \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, error_string);                 \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  } while (0);

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d ''\n",             \
        __FILE__,__LINE__/*,ncclGetErrorString(r)*/);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NVTE_UB_ERROR(x)                                                                           \
  do {                                                                                             \
    throw std::runtime_error(std::string(__FILE__ ":") + std::to_string(__LINE__) +                \
                             " in function " + __func__ + ": " + x);                               \
  } while (false)

int pipe_rank(communicator *comm, int step) {
  int mynode = comm->myrank / comm->nvsize;
  int mylocal = comm->nvrank;
  int numlocal = comm->nvsize;

  int newlocal1 = mylocal + step * comm->ar_nvsize * comm->ar2_nvsize;
  int newlocal = (numlocal + (newlocal1 % numlocal)) % numlocal;
  int newnode = mynode;
  newnode += (newlocal1 - newlocal) / numlocal * comm->num_nodes * comm->num2_nodes;
  int allnodes = comm->nranks / comm->nvsize;
  newnode = (allnodes + (newnode % allnodes)) % allnodes;
  return newnode * numlocal + newlocal;
}

int create_communicator_grouped2(communicator **comm, int pipegpus, int pipenodes, int tensorgpus, int tensornodes
#ifdef NCCLBOOTSTRAP
,ncclComm_t comm_world
#endif
) {
  *comm = reinterpret_cast<communicator *>(malloc(sizeof(communicator)));

  int myrank, nranks, cur_dev, ndev;
 #ifdef NCCLBOOTSTRAP
  #define MPI_MAX_PROCESSOR_NAME 1024
  (*comm)->comm_world=comm_world;
  ncclCommUserRank(comm_world, &myrank);
  ncclCommCount(comm_world, &nranks);
#else
  MPI_Comm_dup(MPI_COMM_WORLD,&(*comm)->comm_world);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
#endif

  (*comm)->nranks = nranks;
  (*comm)->myrank = myrank;
  (*comm)->free_region = 0;
  (*comm)->launch_mode = NVTE_LAUNCH_GPU | NVTE_LAUNCH_CPU;

  cudaDeviceProp device_prop;
  CUDACHECK(cudaGetDevice(&cur_dev));
  CUDACHECK(cudaGetDeviceCount(&ndev));
  CUDACHECK(cudaGetDeviceProperties(&device_prop, cur_dev));
  (*comm)->sm_arch = device_prop.major;
  // (*comm)->use_rr_kernel = device_prop.major == 8;
  (*comm)->use_rr_kernel = 0;
  (*comm)->push = 1;
  (*comm)->use_ce = 0;
  (*comm)->cga_size = 2;
  for (int i = 0; i < userbuffers_op_types; i++)
    (*comm)->basecounter[i] = 0;
  (*comm)->head = 0;
  (*comm)->tail = 0;
  (*comm)->activeproxy = 1;
  (*comm)->active_nreqs = 0;
  for (int i = 0; i < userbuffers_op_types; i++)
    (*comm)->active_req[i].active = -1;

  int ret = 0;
  //split communicator
  int namelen,bytes,color,my_node,mylocal,numlocal,num_nodes;
  int rank=(*comm)->myrank,size=(*comm)->nranks;
  char host_name[MPI_MAX_PROCESSOR_NAME];
  char (*host_names)[MPI_MAX_PROCESSOR_NAME];
  bytes = size * sizeof(char[MPI_MAX_PROCESSOR_NAME]);

#ifdef NCCLBOOTSTRAP
  gethostname(host_name, MPI_MAX_PROCESSOR_NAME);
  cudaMallocHost(&host_names,bytes);
  strcpy(host_names[rank], host_name);
  ncclAllGather(((const void*)host_names)+rank*MPI_MAX_PROCESSOR_NAME, (void*)host_names, MPI_MAX_PROCESSOR_NAME, ncclChar,comm_world, 0); // TODO: check the final 0 for cudaStream_t
  cudaDeviceSynchronize();
#else
 	host_names = (char (*)[MPI_MAX_PROCESSOR_NAME]) malloc(bytes);
  MPI_Get_processor_name(host_name,&namelen);
 	strcpy(host_names[rank], host_name);
 	for (int n=0; n<size; n++)
 		MPI_Bcast(&(host_names[n]),MPI_MAX_PROCESSOR_NAME, MPI_CHAR, n, MPI_COMM_WORLD);
#endif
 	qsort(host_names, size, sizeof(char[MPI_MAX_PROCESSOR_NAME]), stringCmp);
  
 	color = 0;
 	for (int n=0; n<size; n++)  {
   		if(n>0 && strcmp(host_names[n-1], host_names[n])) color++;
   		if(strcmp(host_name, host_names[n]) == 0) break;
 	}

  #ifdef NCCLBOOTSTRAP
    cudaFreeHost(host_names);
    ncclCommSplit(comm_world,color,rank,&(*comm)->comm_intra,NULL);
    ncclCommUserRank((*comm)->comm_intra, &mylocal);
    ncclCommUserCount((*comm)->comm_intra, &numlocal);
  #else
  	free(host_names);
  	MPI_Comm_split(MPI_COMM_WORLD, color, rank, &(*comm)->comm_intra);
  	//find intranode numbers and make internode communicator
    //figure out mylocal
  	MPI_Comm_rank( (*comm)->comm_intra, &mylocal );
    MPI_Comm_size( (*comm)->comm_intra, &numlocal );
  #endif

  (*comm)->nvrank = mylocal;
  (*comm)->nvsize = numlocal;

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  int core;
  if (mylocal == 0)
    core = 50;
  if (mylocal == 1)
    core = 58;
  if (mylocal == 2)
    core = 18;
  if (mylocal == 3)
    core = 26;
  if (mylocal == 4)
    core = 114;
  if (mylocal == 5)
    core = 122;
  if (mylocal == 6)
    core = 82;
  if (mylocal == 7)
    core = 90;

  CPU_SET(core, &cpuset);
  if (!getenv("NVTE_NODOUBLE")) {
    if (core > 128)
      CPU_SET(core - 128, &cpuset);
    else
      CPU_SET(core + 128, &cpuset);
  }
  if (getenv("NVTE_DOPIN"))
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

  if (ndev == numlocal) {  // all visible devices
    if (cur_dev != mylocal)
      printf("%d: device used %d[%d] ,resetting device to %d\n", rank, cur_dev, ndev, mylocal);
    CUDACHECK(cudaSetDevice(mylocal));
  }
  (*comm)->mydev = cur_dev;
  // FIXME need to check that numlocal is multiple of pipegpus x tensorgpus
  // ar1 is data
  int divgpus = pipegpus * tensorgpus;
  int datagpus = numlocal / divgpus;
  (*comm)->ar_nvsize = datagpus;
  (*comm)->ar_firstgpu = mylocal - ((mylocal / tensorgpus) % datagpus) * tensorgpus;
  (*comm)->ar_nvrank = (mylocal - (*comm)->ar_firstgpu) / tensorgpus;
  // ar2 is tensor
  (*comm)->ar2_nvsize = tensorgpus;
  (*comm)->ar2_firstgpu = mylocal - mylocal % tensorgpus;
  (*comm)->ar2_nvrank = mylocal - (*comm)->ar2_firstgpu;
  // ar2 has step equal to ar_nvsize
  int allnodes = nranks / numlocal;
  int mynode = myrank / numlocal;
  int datanodes = allnodes / pipenodes / tensornodes;
  int pipenodegroup_id = myrank / numlocal / (datanodes * tensornodes);

  (*comm)->pipe_id = pipegpus * pipenodegroup_id + mylocal / (datagpus * tensorgpus);
  
  CUDACHECK(cudaFree(0));
  int datanodegroup_id =
      myrank / numlocal / datanodes;  // data reduction group node belongs, equals 0 for all if both
                                      // pipenodes=1 and tensornodes=1
  // mpi communicator only needed for SHARP which is always allreduce1/data-parallel
 #ifdef NCCLBOOTSTRAP
   ncclCommSplit(comm_world,mylocal + numlocal * datanodegroup_id,rank,&(*comm)->comm_inter,NULL);
   ncclCommUserRank((*comm)->comm_inter, &my_node);
   ncclCommUserCount((*comm)->comm_inter, &num_nodes);
 #else
 	MPI_Comm_split(MPI_COMM_WORLD, mylocal + numlocal * datanodegroup_id, rank, &(*comm)->comm_inter);
   //different rails from same group are in different subcommunicators
 
 	MPI_Comm_size( (*comm)->comm_inter, &num_nodes );
 	MPI_Comm_rank( (*comm)->comm_inter, &my_node );
 #endif

  (*comm)->first_node = mynode - my_node;
  (*comm)->num_nodes = num_nodes;
  (*comm)->my_node = my_node;

  (*comm)->num2_nodes = tensornodes;
  (*comm)->my2_node = (mynode / datanodes) % tensornodes;
  (*comm)->first2_node = mynode - (*comm)->my2_node * datanodes;

  char *ib_dev_list;
  int ZIONROCE = getenv("NVTE_ZIONROCE") ? atoi(getenv("NVTE_ZIONROCE")) : 0;
  int ROCE = getenv("NVTE_ROCE") ? atoi(getenv("NVTE_ROCE")) : 0;
  if (ZIONROCE)
    ROCE = 1;
  int DGX_H100 = device_prop.major == 9;

  switch (mylocal) {
  case 0:
    ib_dev_list = "mlx5_0:1";
    break;  // NOLINT(*)
  case 1:
    ib_dev_list = (char *)(DGX_H100 ? "mlx5_3:1" : "mlx5_1:1");  // NOLINT(*)
    break;                                                       // NOLINT(*)
  case 2:
    ib_dev_list = (char *)(ZIONROCE   ? "mlx5_4:1" : DGX_H100 ? "mlx5_4:1" : "mlx5_2:1");  // NOLINT(*)
    break;                                                                                 // NOLINT(*)
  case 3:
    ib_dev_list = (char *)(DGX_H100 ? "mlx5_5:1" : "mlx5_3:1");  // NOLINT(*)
    break;                                                       // NOLINT(*)
  case 4:
    ib_dev_list = (char *)(DGX_H100 ? "mlx5_6:1" : "mlx5_6:1");  // NOLINT(*)
    break;                                                       // NOLINT(*)
  case 5:
    ib_dev_list = (char *)(DGX_H100 ? "mlx5_9:1" : "mlx5_7:1");  // NOLINT(*)
    break;                                                       // NOLINT(*)
  case 6:
    ib_dev_list = (char *)(ZIONROCE   ? "mlx5_10:1" : DGX_H100 ? "mlx5_10:1" : "mlx5_8:1");  // NOLINT(*)
    break;                                                                                   // NOLINT(*)
  case 7:
    ib_dev_list = (char *)(DGX_H100 ? "mlx5_11:1" : "mlx5_9:1");  // NOLINT(*)
    break;                                                        // NOLINT(*)
  default:
    break;
  }

  (*comm)->fifo = reinterpret_cast<ub_request *>(malloc(sizeof(ub_request) * NVTE_MAX_REQUESTS));
  (*comm)->nblocks = 8;
  (*comm)->alignblock = 1024 * 512;
  (*comm)->minblock = 1024 * 2 * 1024;
  (*comm)->asyncblocks = 16;

  CUDACHECK(cudaMallocHost((void **)&(*comm)->hostflags,  // NOLINT(*)
                           (NVTE_MAX_SMS + 100) * sizeof(int)));
  for (int i = 0; i < 100 + NVTE_MAX_SMS; i++)
    (*comm)->hostflags[i] = 0;
  _mm_mfence();
  sleep(1);

  // init_p2p_transport();
  (*comm)->ibnvsize = (*comm)->nvsize;

#define NBUF 2

#define LOCALSIZE 4 * (NVTE_REG0_OFFSET(*comm) + NVTE_REG0_FLAGS + NVTE_REG0_COMMBUFFER * NBUF)
  // peer pointers + op flags + comm buffer

  CUDACHECK(cudaMalloc(&(*comm)->gpu_ptrs,
                       LOCALSIZE));  // flags and pointers, no block data yet
  CUDACHECK(cudaMemset((*comm)->gpu_ptrs, 0, LOCALSIZE));
  CUDACHECK(cudaDeviceSynchronize());
  register_user_buffer_collective(&((*comm)->gpu_ptrs), LOCALSIZE,
                                  *comm);  // will use handler 0
  CUDACHECK(cudaMalloc(&(*comm)->send_id, (*comm)->nranks * sizeof(int)));
  CUDACHECK(cudaMalloc(&(*comm)->recv_id, NVTE_MAX_REGIONS * (*comm)->nranks * sizeof(int)));
  CUDACHECK(cudaMemset((*comm)->send_id, 0, (*comm)->nranks * sizeof(int)));
  CUDACHECK(cudaMemset((*comm)->recv_id, 0, NVTE_MAX_REGIONS * (*comm)->nranks * sizeof(int)));
  (*comm)->sms = 16;
  (*comm)->threads = 1024;

#define GPU_PAGE_SHIFT 16
#define GPU_PAGE_SIZE (1UL << GPU_PAGE_SHIFT)
#define GPU_PAGE_OFFSET (GPU_PAGE_SIZE - 1)
#define GPU_PAGE_MASK (~GPU_PAGE_OFFSET)
  CUDACHECK(cudaMalloc(&(*comm)->flags, 2 * GPU_PAGE_SIZE));
  unsigned int flag = 1;
  CUDACHECK(cudaMemset((*comm)->flags, 0, 2 * GPU_PAGE_SIZE));
  (*comm)->flags =
      reinterpret_cast<int *>(((CUdeviceptr)(*comm)->flags + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK);

  using namespace std;
  (*comm)->g = gdr_open();
  if ((*comm)->g == NULL) {
    fprintf(stderr, "gdrcopy open failed\n");
    return -1;
  }
  gdr_mh_t mh;
  ret = gdr_pin_buffer((*comm)->g, (CUdeviceptr)(*comm)->flags, GPU_PAGE_SIZE, 0, 0, &mh);
  if (ret) {
    fprintf(stderr, "gdr_pin_buffer failed\n");
    return -1;
  }
  ret = gdr_map((*comm)->g, mh, (void **)&((*comm)->map_flags), GPU_PAGE_SIZE);  // NOLINT(*)

  if (ret) {
    fprintf(stderr, "gdr_map failed\n");
    return -1;
  }
  sched_param param;
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_getschedparam(&attr, &param);
  param.sched_priority = sched_get_priority_max(SCHED_FIFO);

  pthread_attr_setschedparam(&attr, &param);

  if (getenv("NVTE_UBDEBUG"))
    printf("%d/%d:(%d x %d): DP %d x %d TP %d x %d, DPGROUP %dx%d TPGROUP "
           "%dx%d PIPE_ID %d/%d\n",
           myrank, nranks, myrank / numlocal, myrank % numlocal, (*comm)->my_node,
           (*comm)->ar_nvrank, (*comm)->my2_node, (*comm)->ar2_nvrank, (*comm)->num_nodes,
           (*comm)->ar_nvsize, (*comm)->num2_nodes, (*comm)->ar2_nvsize, (*comm)->pipe_id,
           pipegpus * pipenodes);
  fflush(NULL);

  return 0;
}

int create_communicator_grouped( communicator** comm, int pipegpus, int pipenodes
#ifdef NCCLBOOTSTRAP
,ncclComm_t comm_world
#endif
) 
{ return create_communicator_grouped2(comm,pipegpus,pipenodes,1,1
#ifdef NCCLBOOTSTRAP
,comm_world
#endif
); }

int create_communicator( communicator** comm 
#ifdef NCCLBOOTSTRAP
,ncclComm_t comm_world
#endif
) {
  return create_communicator_grouped2(comm,1,1,1,1
#ifdef NCCLBOOTSTRAP
,comm_world
#endif
); }

void destroy_communicator(communicator *comm) {
  comm->activeproxy = 0;
  if (!comm->myrank && getenv("NVTE_UBDEBUG"))
    printf("waiting for userbuffers proxy thread to exit()\n");
  gdr_close(comm->g);
}

int register_user_buffer_collective(void **gpubuff, size_t bytes, communicator *comm, bool alloc) {
  if (comm->free_region > NVTE_MAX_REGIONS)
    return -1;
  int hndl = comm->free_region;
  comm->peer_ptr[hndl] = reinterpret_cast<void **>(malloc(sizeof(void *) * (comm->nvsize)));

  if (alloc) {
    CUDACHECK(cudaMalloc(gpubuff, bytes));
  }
  assert(comm->nvsize <= 8);
  cudaIpcMemHandle_t *memhndl =
      reinterpret_cast<cudaIpcMemHandle_t *>(malloc(sizeof(cudaIpcMemHandle_t) * (comm->nvsize)));

  CUDACHECK(cudaIpcGetMemHandle(&memhndl[comm->nvrank], *gpubuff));

  #ifdef NCCLBOOTSTRAP
    // Allocate device memory for the IPC memory handles
    cudaIpcMemHandle_t *d_memhndl;
    CUDACHECK(cudaMalloc(&d_memhndl, comm->nvsize * sizeof(cudaIpcMemHandle_t)));

    // Copy the local IPC memory handle to device memory
    CUDACHECK(cudaMemcpy(&d_memhndl[comm->nvrank], &memhndl[comm->nvrank], sizeof(cudaIpcMemHandle_t), cudaMemcpyHostToDevice));

    // Perform the all-gather operation
    NCCLCHECK(ncclAllGather(&d_memhndl[comm->nvrank], d_memhndl, sizeof(cudaIpcMemHandle_t) / sizeof(int8_t), ncclInt8, comm->comm_intra, 0));

    // Copy the gathered IPC memory handles back to host memory
    CUDACHECK(cudaMemcpy(memhndl, d_memhndl, comm->nvsize * sizeof(cudaIpcMemHandle_t), cudaMemcpyDeviceToHost));

    // Free the device memory
    CUDACHECK(cudaFree(d_memhndl));
  #else 
  MPI_Allgather(&memhndl[comm->nvrank], sizeof(cudaIpcMemHandle_t), MPI_BYTE, memhndl,
                sizeof(cudaIpcMemHandle_t), MPI_BYTE, comm->comm_intra);
  #endif
  
  for (int i = 0; i < comm->nvsize; i++)
    if (i != comm->nvrank)
      CUDACHECK(cudaIpcOpenMemHandle((void **)&(comm->peer_ptr[hndl][i]),  // NOLINT(*)
                                     memhndl[i], cudaIpcMemLazyEnablePeerAccess));
  comm->peer_ptr[hndl][comm->nvrank] = *gpubuff;
  CUDACHECK(cudaDeviceSynchronize());
  CUDACHECK(
      cudaMemcpy(reinterpret_cast<char *>(comm->gpu_ptrs) + (hndl * comm->nvsize * sizeof(void *)),
                 comm->peer_ptr[hndl], comm->nvsize * sizeof(void *), cudaMemcpyHostToDevice));
  CUDACHECK(cudaDeviceSynchronize());
  free(memhndl);

  comm->mem_ptr[hndl] = *gpubuff;

  return comm->free_region++;
}

int allreduce_userbuff_inplace_gpu(const int handler, const int offset, const int elements,
                                   const int blocksize, communicator *comm, cudaStream_t stream);

int allreduce2_userbuff_inplace_gpu(const int maxcredit, const int handler, const int offset,
                                    const int elements, const int blocksize, communicator *comm,
                                    cudaStream_t stream, int op);

int reducescatter2_userbuff_inplace_gpu(const int maxcredit, const int handler, const int offset,
                                        const int elements, const int blocksize, communicator *comm,
                                        cudaStream_t stream, int op);

int allgather2_userbuff_inplace_gpu(const int maxcredit, const int handler, const int offset,
                                    const int elements, const int blocksize, communicator *comm,
                                    cudaStream_t stream, int op);

void allreduce_nonsharp_inplace(const int handler, const int offset, const int elements,
                                communicator *comm, cudaStream_t stream, int op) {
  if (elements < 64)
    NVTE_UB_ERROR("Userbuffer comm for given config not implemented.");
  // if(comm->myrank==0) fprintf(stderr,"AR2(%d) user call
  // launch_mode=%d\n",op,comm->launch_mode);
  const int ar_nvsize = op == userbuffers_allreduceop_nonsharp ? comm->ar_nvsize : comm->ar2_nvsize;
  int blocksize = elements * 2;
  int maxcredit = 0;
  const int num_nodes = op == userbuffers_allreduceop_nonsharp ? comm->num_nodes : comm->num2_nodes;
  blocksize = (comm->nblocks - 1 + (comm->alignblock - 1 + elements * 2) / comm->alignblock) /
              comm->nblocks;  // FIXME TUNING
  blocksize *= comm->alignblock;
  if (blocksize < comm->minblock)
    blocksize = comm->minblock;

  maxcredit = (elements * 2 + blocksize - 1) / blocksize;
  size_t peerblock = sizeof(int) * NVTE_REG0_COMMBUFFER / maxcredit;  // max size we can fit
  if (blocksize > peerblock * ar_nvsize)
    blocksize = peerblock * ar_nvsize;
  int sms = allreduce2_userbuff_inplace_gpu(maxcredit, handler, offset, elements, blocksize, comm,
                                            stream, op);

  if (num_nodes > 1 && comm->launch_mode & NVTE_LAUNCH_CPU) {
    if (!sms)
      return;
    comm->fifo[comm->head].optype = op;
    comm->fifo[comm->head].basecounter = comm->basecounter[op];
    comm->fifo[comm->head].blocksize = blocksize;
    comm->fifo[comm->head].maxcredit = maxcredit;
    comm->fifo[comm->head].handler = handler;
    comm->fifo[comm->head].offset = offset;
    comm->fifo[comm->head].elements = elements;

    int newhead = (comm->head + 1) & (NVTE_MAX_REQUESTS - 1);
    while (newhead == comm->tail) {
    }
    comm->head = newhead;

    comm->basecounter[op] += (elements * 2 + blocksize - 1) / blocksize;
  }
}

void allreduce2_userbuff_inplace(const int handler, const int offset, const int elements,
                                 communicator *comm, cudaStream_t stream) {
  allreduce_nonsharp_inplace(handler, offset, elements, comm, stream,
                             userbuffers_allreduceop_nonsharp2);
}

void allreduce_userbuff_inplace(const int handler, const int offset, const int elements,
                                communicator *comm, cudaStream_t stream) {
  if (elements < 64)
    NVTE_UB_ERROR("Userbuffer comm for given config not implemented.");
  allreduce_nonsharp_inplace(handler, offset, elements, comm, stream,
                             userbuffers_allreduceop_nonsharp);
  return;
}

void reducescatter_userbuff_inplace(const int handler, const int offset, const int elements,
                                    communicator *comm, cudaStream_t stream) {
  if (elements < 64)
    NVTE_UB_ERROR("Userbuffer comm for given config not implemented.");

  int op = userbuffers_allreduceop_nonsharp;
  const int ar_nvsize = op == userbuffers_allreduceop_nonsharp ? comm->ar_nvsize : comm->ar2_nvsize;
  int blocksize = elements * 2;
  int maxcredit = 0;

  const int num_nodes = op == userbuffers_allreduceop_nonsharp ? comm->num_nodes : comm->num2_nodes;
  blocksize = (comm->nblocks - 1 + (comm->alignblock - 1 + elements * 2) / comm->alignblock) /
              comm->nblocks;  // FIXME TUNING
  blocksize *= comm->alignblock;
  if (blocksize < comm->minblock)
    blocksize = comm->minblock;

  maxcredit = (elements * 2 + blocksize - 1) / blocksize;
  size_t peerblock = sizeof(int) * NVTE_REG0_COMMBUFFER / maxcredit;  // max size we can fit
  if (blocksize > peerblock * ar_nvsize)
    blocksize = peerblock * ar_nvsize;

  int sms = reducescatter2_userbuff_inplace_gpu(maxcredit, handler, offset, elements, blocksize,
                                                comm, stream, op);

  if (num_nodes > 1 && comm->launch_mode & NVTE_LAUNCH_CPU) {
    if (!sms)
      return;
    comm->fifo[comm->head].optype = op;
    comm->fifo[comm->head].basecounter = comm->basecounter[op];
    comm->fifo[comm->head].blocksize = blocksize;
    comm->fifo[comm->head].maxcredit = maxcredit;
    comm->fifo[comm->head].handler = handler;
    comm->fifo[comm->head].offset = offset;
    comm->fifo[comm->head].elements = elements;

    int newhead = (comm->head + 1) & (NVTE_MAX_REQUESTS - 1);
    while (newhead == comm->tail) {
    }
    comm->head = newhead;

    comm->basecounter[op] += (elements * 2 + blocksize - 1) / blocksize;
  }
}

void allgather_userbuff_inplace(const int handler, const int offset, const int elements,
                                communicator *comm, cudaStream_t stream) {
  if (elements < 64)
    NVTE_UB_ERROR("Userbuffer comm for given config not implemented.");
  int op = userbuffers_allreduceop_nonsharp;
  const int ar_nvsize = op == userbuffers_allreduceop_nonsharp ? comm->ar_nvsize : comm->ar2_nvsize;
  int blocksize = elements * 2;
  int maxcredit = 0;

  const int num_nodes = op == userbuffers_allreduceop_nonsharp ? comm->num_nodes : comm->num2_nodes;
  blocksize = (comm->nblocks - 1 + (comm->alignblock - 1 + elements * 2) / comm->alignblock) /
              comm->nblocks;  // FIXME TUNING
  blocksize *= comm->alignblock;
  if (blocksize < comm->minblock)
    blocksize = comm->minblock;

  maxcredit = (elements * 2 + blocksize - 1) / blocksize;
  size_t peerblock = sizeof(int) * NVTE_REG0_COMMBUFFER / maxcredit;  // max size we can fit
  if (blocksize > peerblock * ar_nvsize)
    blocksize = peerblock * ar_nvsize;

  int sms = allgather2_userbuff_inplace_gpu(maxcredit, handler, offset, elements, blocksize, comm,
                                            stream, op);
}
