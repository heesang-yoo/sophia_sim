void ISPH(int_t*vii,Real*vif)
{
	//-------------------------------------------------------------------------------------------------
	// GPU device properties
	//-------------------------------------------------------------------------------------------------
	struct cudaDeviceProp prop;
	{
		int_t gcount,i;
		cudaGetDeviceCount(&gcount);

		for(i=0;i<gcount;i++){
			cudaGetDeviceProperties(&prop,i);
			printf("### GPU DEVICE PROPERTIES.................................\n\n");
			printf("	Name: %s\n",prop.name);
			printf("	Compute capability: %d.%d\n",prop.major,prop.minor);
			printf("	Clock rate: %d\n",prop.clockRate);
			printf("	Total global memory: %ld\n",prop.totalGlobalMem);
			printf("	Total constant memory: %d\n",prop.totalConstMem);
			printf("	Multiprocessor count: %d\n",prop.multiProcessorCount);
			printf("	Shared mem per block: %d\n",prop.sharedMemPerBlock);
			printf("	Registers per block: %d\n",prop.regsPerBlock);
			printf("	Threads in warp: %d\n",prop.warpSize);
			printf("	Max threads per block: %d\n",prop.maxThreadsPerBlock);
			printf("	Max thread dimensions: %d,%d,%d\n",prop.maxThreadsDim[0],prop.maxThreadsDim[1],prop.maxThreadsDim[2]);
			printf("	Max grid dimensions: %d,%d,%d\n",prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);
			printf("...........................................................\n\n");
		}
	}
	printf(" ------------------------------------------------------------\n");
	printf(" SOPHIA_gpu v.1.0 \n");
	printf(" Developed by E.S. Kim,Y.B. Jo,S.H. Park\n");
	printf(" 2017. 02. 20 \n");
	printf(" Optimized by Y.W. Sim, CoCoLink Inc.\n");
	printf(" 2018, 2019(C) \n");
	printf(" Restructured & Innovated by Eung Soo Kim, Hee Sang Yoo, Young Beom Jo, Hae Yoon Choi, Su-San Park, Jin Woo Kim, Yelyn Ahn, Tae Soo Choi\n");
	printf(" ESLAB, SEOUL NATIONAL UNIVERSITY, SOUTH KOREA.\n");
	printf(" 2019. 08. 08 \n");
	printf("------------------------------------------------------------\n\n");
	//-------------------------------------------------------------------------------------------------


	//-------------------------------------------------------------------------------------------------
	// 입출력 파일 이름 설정 (초기화)
	//-------------------------------------------------------------------------------------------------

	char INPUT_FILE_NAME[128];
	strcpy(INPUT_FILE_NAME,"./input/input.txt");								// input file name and address

	// char MARKER_INPUT_FILE_NAME[128];
	// strcpy(MARKER_INPUT_FILE_NAME,"./input/marker.txt");

	//-------------------------------------------------------------------------------------------------
	// 입자 개수 세기
	//-------------------------------------------------------------------------------------------------

	num_part=gpu_count_particle_numbers2(INPUT_FILE_NAME);
	// num_marker=gpu_count_particle_numbers2(MARKER_INPUT_FILE_NAME);


	//-------------------------------------------------------------------------------------------------
	// 입력파일 읽고 host 입자 (HP1)에 정보 저장
	//-------------------------------------------------------------------------------------------------

	// host 입자 (HP1) 메모리 할당 & 초기화
	HP1=(part1*)malloc(num_part*sizeof(part1));
	memset(HP1,0,sizeof(part1)*num_part);


	// 입력파일 (input.txt) 읽기
	read_input(HP1);


	//-------------------------------------------------------------------------------------------------
	// 셀 (Cell) 및 검색 범위 설정
	//-------------------------------------------------------------------------------------------------

	// 검색범위 관련 변수 (default)
	Real cell_reduction_factor=1.1;
	search_incr_factor=1.0;											// coefficient for cell and search range (esk)

	search_kappa=kappa;


	//-------------------------------------------------------------------------------------------------
	// 전체 계산 영역 파악 및 셀(Cell) 분할
	//-------------------------------------------------------------------------------------------------

	// 계산범위
	find_minmax(vii,vif,HP1);

	// 셀 간격(dcell)
	Real h0=h_max;
	dcell=cell_reduction_factor*kappa*h0/ncell_init;

  // 셀 개수(NI, NJ, NK)
	NI=(int)((x_max-x_min)/dcell)+1;
	NJ=(int)((y_max-y_min)/dcell)+1;
	NK=(int)((z_max-z_min)/dcell)+1;

	int tNx=(int)((x_max-x_min)/(h0/h_coeff))+1;
	int tNz=(int)((z_max-z_min)/(h0/h_coeff))+1;

	//-------------------------------------------------------------------------------------------------
	// GPU 당 셀 개수 및 필요한 입자 메모리 크기 계산
	//-------------------------------------------------------------------------------------------------

	// 각 GPU x 방향 셀 개수 (NI를 총 gpu 개수로 나눔)
	calc_area=ceil(NI/ngpu);

	// 각 GPU 당 입자 정보를 할당할 메모리 크기 설정
	if(ngpu>1){
		num_p2p=(int)(num_part*4/NI*C_p2p);								// (CAUTION)
		num_part2=(int)((num_part/ngpu)*1.2)+2*num_p2p;		// (CAUTION)
	}else{
		if(open_boundary>0)
		{
			space=h0/h_coeff*0.5;
			printf("\n\n----------------------------------------------------\n");
			printf("Open Boundary Space=%f \n", space);
			Nsx=(int)((x_max-x_min)/space)+1;		// (CAUTION)
			// Nsz=(int)((z_max-z_min)/space)+1;		// (CAUTION)
			Nsz=(int)((y_max-y_min)/space)+1;		// (CAUTION)

			// if (dim==2) buffer_size=(Nsx)*60.0;
			// if (dim==3) buffer_size=(Nsx*Nsz)*60.0;

			buffer_size=(Nsx*Nsz);
		}
		// if(open_boundary>0)
		// // {
		// 	space=h0/h_coeff*0.5;
		// 	Nsx=(int)((x_max-x_min)/space)+1;		// (CAUTION)
		// 	// Nsy=(int)((y_max-y_min)/space)+1;		// (CAUTION)
		// 	// Nsz=(int)((z_max-z_min)/space)+1;		// (CAUTION)

		// 	buffer_size=Nsx*100;
		// 	// }
		// 	printf("\n\n----------------------------------------------------\n");
		// 	printf("buffer for mirror particles = %d \n",buffer_size);


			buffer_size=0;
			
		num_part2=num_part+buffer_size;
		num_part3=num_part2;

		if(aps_solv){
			if(dim==2) {
				num_part2=num_part+num_part*4;
				num_part3=num_part2+buffer_size;
			}
			if(dim==3) num_part3=4*num_part+buffer_size;
		}

		if(open_boundary>0)
		{
			if (dim==2) printf("buffer memory size: %d	ratio: %.2f \n\n", buffer_size, (Real) buffer_size/tNx);
			if (dim==3) printf("buffer memory size: %d	ratio: %.2f \n\n", buffer_size, (Real) buffer_size/(tNx*tNz));
			printf("frequency of inlet particle generation : %d step\n", (int) ((h0/(h_coeff))/(Inlet_Velocity*dt)));
			printf("----------------------------------------------------\n");

		}

	}


	//-------------------------------------------------------------------------------------------------
	// P2P 활성화
	//-------------------------------------------------------------------------------------------------


	if(ngpu>1){
		for(int i=0;i<ngpu;i++){
			cudaSetDevice(i);
			for(int j=0;j<ngpu;j++){
				cudaDeviceEnablePeerAccess(j,0);
			}
		}
	}


	//-------------------------------------------------------------------------------------------------
	// 코드 실행
	//-------------------------------------------------------------------------------------------------

	// WCSPH Dev Calc Function using Pthread
	pthread_t*solve_thread;
	void*thread_result;
	int*tid;

	tid=(int*)malloc(sizeof(int)*ngpu);
	solve_thread=(pthread_t*)malloc(sizeof(pthread_t)*ngpu);
	pthread_barrier_init(&barrier,NULL,ngpu);

	//WCSPH_Calc();
	for(int i=0;i<ngpu;i++){
		tid[i]=0;
		pthread_create(&solve_thread[i],NULL,ISPH_Calc,(void*)&tid[i]);
	}

	for(int i=0;i<ngpu;i++) pthread_join(solve_thread[i],&thread_result);
	pthread_barrier_destroy(&barrier);

	if(ngpu>1){
		for(int i=0;i<ngpu;i++){
			cudaSetDevice(i);
			for(int j=0;j<ngpu;j++){
				cudaDeviceDisablePeerAccess(j);
			}
		}
	}

	free(HP1);
	free(solve_thread);
	free(tid);
}
