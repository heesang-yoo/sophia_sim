//-------------------------------------------------------------------------------------------------
// 모든 입자 계산을 수행하는 주함수
//-------------------------------------------------------------------------------------------------
void SOPHIA_single_ISPH(int_t*g_idx,int_t*p_idx,int_t*g_idx_in,int_t*p_idx_in,int_t*g_str,int_t*g_end,
	part1*dev_P1,part1*dev_SP1,part2*dev_P2,part2*dev_SP2,part3*dev_P3,
	int_t*p2p_af_in,int_t*p2p_idx_in,int_t*p2p_af,int_t*p2p_idx,
	void*dev_sort_storage,size_t*sort_storage_bytes,part1*file_P1,part2*file_P2,part3*file_P3,int tid
	,Real*time0,Real*P0,Real*P1,Real*P2,Real*P3,int_t*plotcount)
{

dim3 b,t;
t.x=128;
b.x=(num_part3-1)/t.x+1;
int s=sizeof(int)*(t.x+1);


KERNEL_set_ncell<<<b,t>>>(dev_P1,count);
cudaDeviceSynchronize();

if(scheme==Lagrangian){
KERNEL_set_alpha_Lagrangian<<<b,t>>>(dev_P1);
cudaDeviceSynchronize();
}else if(scheme==ALE){
KERNEL_set_alpha<<<b,t>>>(dev_P1);
}


//-------------------------------------------------------------------------------------------------
// 주변입자 검색
//-------------------------------------------------------------------------------------------------

if(((scheme==Eulerian)&&(count==0))||(scheme>Eulerian)){

// g_str을 리셋
cudaMemset(g_str,cu_memset,sizeof(int_t)*num_cells);

// 입자의 셀번호 계산
b.x=(num_part3-1)/t.x+1;
KERNEL_index_particle_to_cell<<<b,t>>>(g_idx_in,p_idx_in,dev_P1);
cudaDeviceSynchronize();

// 셀번호를 바탕으로 정렬
cub::DeviceRadixSort::SortPairs(dev_sort_storage,*sort_storage_bytes,g_idx_in,g_idx,p_idx_in,p_idx,num_part3);
cudaDeviceSynchronize();

// 정렬한 입자를 재배치
b.x=(num_part3-1)/t.x+1;
KERNEL_reorder<<<b,t,s>>>(g_idx,p_idx,g_str,g_end,dev_P1,dev_P2,dev_SP1,dev_SP2);
cudaDeviceSynchronize();

// 일부 입자정보 리셋
// cudaMemset(dev_P3,0,sizeof(part3)*num_part3);

// 입자정보를 P1 에 복사
cudaMemcpy(dev_P1,dev_SP1,sizeof(part1)*num_part3,cudaMemcpyDeviceToDevice);


}else if((scheme==Eulerian)&&(count>0)){

// 일부 입자정보 리셋
// cudaMemset(dev_P3,0,sizeof(part3)*num_part3);

// 입자정보를 P1 에 복사
cudaMemcpy(dev_SP1,dev_P1,sizeof(part1)*num_part3,cudaMemcpyDeviceToDevice);

cudaMemcpy(dev_SP2,dev_P2,sizeof(part2)*num_part3,cudaMemcpyDeviceToDevice);

}

if(multi_type==2||multi_type==4){
	KERNEL_smoothing_length2D<<<b,t>>>(g_str,g_end,dev_SP1);
	cudaDeviceSynchronize();
	}
	
//-------------------------------------------------------------------------------------------------
// 압력힘을 제외한 힘 계산
//-------------------------------------------------------------------------------------------------

// Velocity condition for wall
b.x=(num_part3-1)/t.x+1;
if((noslip_bc==1)||(penetration_solve==1)){
	if(dim==2) KERNEL_boundary2D<<<b,t>>>(g_str,g_end,dev_SP1);
	if(dim==3) KERNEL_boundary3D<<<b,t>>>(g_str,g_end,dev_SP1);
	cudaDeviceSynchronize();
}
		
b.x=(num_part3-1)/t.x+1;
if(dim==2) KERNEL_advection_force2D<<<b,t>>>(time,1,g_str,g_end,dev_SP1,dev_SP2,dev_P3);
if(dim==3) KERNEL_advection_force3D<<<b,t>>>(1,g_str,g_end,dev_SP1,dev_SP2,dev_P3);
cudaDeviceSynchronize();

//-------------------------------------------------------------------------------------------------
// PREDICTOR (Optional)
//-------------------------------------------------------------------------------------------------

if(time_type==Pre_Cor){
b.x=(num_part3-1)/t.x+1;
KERNEL_clc_projection<<<b,t>>>(count,dt,time,dev_SP1,dev_SP2,dev_P3);
cudaDeviceSynchronize();

// IBM_predictor<<<b,t>>>(dt_structure,time,dev_SP1,dev_SP2);
// cudaDeviceSynchronize();
}

//-------------------------------------------------------------------------------------------------
// 계산 준비: gradient correction, filter, reference density, p_type switch, penetration, density gradient
//-------------------------------------------------------------------------------------------------

b.x=(num_part3-1)/t.x+1;
// 미분보정필터 계산
gradient_correction(g_str,g_end,dev_SP1,dev_P3);
cudaDeviceSynchronize();
// if(dim==2) KERNEL_clc_correction_preLaplacian<<<b,t>>>(g_str,g_end,dev_SP1,dev_P3);
// cudaDeviceSynchronize();
// if(dim==2) KERNEL_clc_correction_Laplacian<<<b,t>>>(g_str,g_end,dev_SP1,dev_P3);
// cudaDeviceSynchronize();
// KERNEL_clc_invA_MLS_2D<<<b,t>>>(g_str,g_end,dev_SP1,dev_P3);
// cudaDeviceSynchronize();

// filter, reference density, p_type switch, penetration, normal gradient etc
if(dim==2) KERNEL_clc_prep2D<<<b,t>>>(g_str,g_end,dev_SP1, dev_SP2, dev_P3, count);
if(dim==3) KERNEL_clc_prep3D<<<b,t>>>(g_str,g_end,dev_SP1, dev_SP2, dev_P3, count);
cudaDeviceSynchronize();

//-------------------------------------------------------------------------------------------------
// 표면입자 추출
//-------------------------------------------------------------------------------------------------

// Detection
// if(dim==2)	KERNEL_clc_surface_detect2D<<<b,t>>>(g_str,g_end,dev_SP1,dev_P3);
// cudaDeviceSynchronize();

//-------------------------------------------------------------------------------------------------
// 압력 계산
//-------------------------------------------------------------------------------------------------

// For initial condition
if(count==0){
	if(dim==2)	KERNEL_Neumann_boundary2D<<<b,t>>>(g_str,g_end,dev_SP1,dev_SP2,dev_P3);
	if(dim==3)	KERNEL_Neumann_boundary3D<<<b,t>>>(g_str,g_end,dev_SP1,dev_SP2,dev_P3);
cudaDeviceSynchronize();
}

for(int_t it=0;it<=0;it++){
// PPE
if(dim==2)	KERNEL_PPE2D<<<b,t>>>(dt,g_str,g_end,dev_SP1,dev_SP2,dev_P3,it);
if(dim==3)	KERNEL_PPE3D<<<b,t>>>(dt,g_str,g_end,dev_SP1,dev_SP2,dev_P3);
cudaDeviceSynchronize();

// Pressure condition for wall
if(dim==2)	KERNEL_Neumann_boundary2D<<<b,t>>>(g_str,g_end,dev_SP1,dev_SP2,dev_P3);
if(dim==3)	KERNEL_Neumann_boundary3D<<<b,t>>>(g_str,g_end,dev_SP1,dev_SP2,dev_P3);
cudaDeviceSynchronize();
}

//-------------------------------------------------------------------------------------------------
// 압력힘 계산
//-------------------------------------------------------------------------------------------------

	b.x=(num_part3-1)/t.x+1;
	if(dim==2) KERNEL_pressureforce2D<<<b,t>>>(1,g_str,g_end,dev_SP1,dev_SP2,dev_P3);
	if(dim==3) KERNEL_pressureforce3D<<<b,t>>>(1,g_str,g_end,dev_SP1,dev_SP2,dev_P3);
	cudaDeviceSynchronize();


//-------------------------------------------------------------------------------------------------
// Immersed Boundary Method (IBM)
//-------------------------------------------------------------------------------------------------

// b.x=(num_part3-1)/t.x+1;
// if(dim==2) IBM_force_interpolation<<<b,t>>>(dt,g_str,g_end,dev_SP1,dev_SP2,dev_P3);
// cudaDeviceSynchronize();


// if(dim==2) IBM_spreading_interpolation<<<b,t>>>(g_str,g_end,dev_SP1,dev_SP2,dev_P3);
// cudaDeviceSynchronize();

//-------------------------------------------------------------------------------------------------
// 시간 적분 (Time Integration)
//-------------------------------------------------------------------------------------------------

b.x=(num_part3-1)/t.x+1;
KERNEL_time_update_projection<<<b,t>>>(dt,dev_SP1,dev_P1,dev_SP2,dev_P2,dev_P3);
cudaDeviceSynchronize();

// IBM_corrector<<<b,t>>>(dt_structure,time,dev_SP1,dev_P1,dev_SP2,dev_P3);
// cudaDeviceSynchronize();

//-------------------------------------------------------------------------------------------------
// Open Boudnary
//-------------------------------------------------------------------------------------------------

// if(count%(lap)==0){

// b.x=(num_part3-1)/t.x+1;
// if(open_boundary>0)
// {
// if(dim==2){
// KERNEL_open_boundary_extrapolation2D<<<b,t>>>(time, g_str,g_end,dev_P1,dev_SP2,dev_P3,count,dt);
// cudaDeviceSynchronize();
// }

// KERNEL_time_update_buffer<<<b,t>>>(dt,dev_SP1,dev_P1,dev_SP2,dev_P3,space,Nsx,Nsz);
// cudaDeviceSynchronize();
// }


// if(dim==2) KERNEL_clc_normalvectorforpst<<<b,t>>>(g_str,g_end,dev_P1, dev_SP2, dev_P3, count);
// cudaDeviceSynchronize();

	// device
	Real*max_umag,*d_max_umag0;
	cudaMalloc((void**)&max_umag,sizeof(Real)*num_part3);
	cudaMalloc((void**)&d_max_umag0,sizeof(Real));
	cudaMemset(max_umag,0,sizeof(Real)*num_part3);
	cudaMemset(d_max_umag0,0,sizeof(Real));

	// Sorting & Max variable to use CUB Library
	void*dev_max_storage=NULL;
	size_t max_storage_bytes=0;

	// Determine Sorting & Maximum Value Setting for Total Particle Data
	cub::DeviceReduce::Max(dev_max_storage,max_storage_bytes,max_umag,d_max_umag0,num_part3);
	cudaDeviceSynchronize();
	cudaMalloc((void**)&dev_max_storage,max_storage_bytes);
	

	kernel_copy_max_velocity<<<b,t>>>(dev_P1,dev_SP2,dev_P3,max_umag);
	cudaDeviceSynchronize();

	// Find Max Velocity & Force using CUB - TID=0
	cub::DeviceReduce::Max(dev_max_storage,max_storage_bytes,max_umag,d_max_umag0,num_part3);
	cudaDeviceSynchronize();

	b.x=(num_part3-1)/t.x+1;
	if(pst_solve==1){
	calculate_w_dx(dev_P1);
	cudaDeviceSynchronize();
	// if(dim==2) KERNEL_clc_particle_shifting_lind2D<<<b,t>>>(g_str,g_end,dev_P1,dev_SP2,dev_P3,dt);
	if(dim==2) KERNEL_clc_particle_shifting_oger2D<<<b,t>>>(g_str,g_end,dev_P1,dev_SP2,dev_P3,dt,d_max_umag0);
	// if(dim==2) KERNEL_clc_particle_shifting_oger2D2<<<b,t>>>(g_str,g_end,dev_P1,dev_SP2,dev_P3,dt,d_max_umag0);
	// if(dim==3) KERNEL_clc_particle_shifting_lind3D<<<b,t>>>(g_str,g_end,dev_P1,dev_SP2,dev_P3,dt);
	cudaDeviceSynchronize();
	// if(dim==2) KERNEL_clc_particle_shifting_lind2D2<<<b,t>>>(g_str,g_end,dev_P1,dev_SP2,dev_P3,dt);
	// cudaDeviceSynchronize();
	}

	cudaFree(d_max_umag0);
	cudaFree(max_umag);

// //-------------------------------------------------------------------------------------------------
// // particle splitting/merging
// //-------------------------------------------------------------------------------------------------

// if((aps_solv)&(count%100==0)) //임시로함.
// {
// 	int APS_num[5]={0,0,0,0,0};
// 	int*dev_APS_num;
// 	cudaMalloc((void**)&dev_APS_num,5*sizeof(int));
// 	cudaMemset(dev_APS_num,0,5*sizeof(int));

// 	KERNEL_reset_APS_variables<<<b,t>>>(dev_P1);
// 	cudaDeviceSynchronize();

// 	b.x=(num_part3-1)/t.x+1;
// 	if(dim==2) KERNEL_APS_condition2D<<<b,t>>>(dt,time,dev_P1,dev_APS_num,g_str,g_end,aps);
// 	// if(dim==3) KERNEL_APS_condition3D<<<b,t>>>(dev_P1,dev_APS_num,g_str,g_end);
// 	cudaDeviceSynchronize();

// 	cudaMemcpy(APS_num,dev_APS_num,5*sizeof(int),cudaMemcpyDeviceToHost);

// 	if(count%freq_output==0){
// 		printf("number of splitting particles = %d \n",APS_num[1]);
// 		printf("number of merging particles = %d \n",APS_num[2]);
// 	}

// 		if(APS_num[0]==1){                                   //calculating APS

// 		int*mutex;
// 		cudaMalloc((void**)&mutex,sizeof(int));
// 		cudaMemset(mutex,0,sizeof(int));
// 		// if(APS_num[1]>0){                                 //particle splitting

// 		// if(APS_num[1]>1) t.x=1;
// 		// else t.x=128;
// 		t.x=1;
// 		b.x=(num_part3-1)/t.x+1;
// 		KERNEL_assign_split_num<<<b,t>>>(dev_P1,dev_APS_num,mutex);
// 		cudaDeviceSynchronize();

// 		t.x=1;
// 		b.x=(num_part3-1)/t.x+1;
// 		if(dim==2) KERNEL_particle_splitting2D<<<b,t>>>(count,dev_P1,aps_num_part,mutex);
// 		// if(dim==3) KERNEL_particle_splitting3D<<<b,t>>>(dev_P1,aps_num_part);
// 		cudaDeviceSynchronize();

// 		// }

// 		//  if(APS_num[2]>=1){    															 //particle merging
// 		// 	t.x=1;
// 		// 	b.x=(num_part3-1)/t.x+1;
// 		// 	KERNEL_merge_or_not<<<b,t>>>(dev_P1,g_str,g_end,mutex);
// 		// 	cudaDeviceSynchronize();

// 		// 	// if(APS_num[2]>1) t.x=1;
// 		// 	// else t.x=128;
// 		// 	t.x=1;
// 		// 	b.x=(num_part3-1)/t.x+1;
// 		// 	if(dim==2) KERNEL_assign_merge_num2D<<<b,t>>>(dev_P1,g_str,g_end,dev_APS_num,mutex,aps);
// 		// 	// if(dim==3) KERNEL_assign_merge_num3D<<<b,t>>>(dev_P1,g_str,g_end,dev_APS_num,mutex);
// 		// 	cudaDeviceSynchronize();

// 		// 	// cudaMemcpy(APS_num,dev_APS_num,5*sizeof(int),cudaMemcpyDeviceToHost);
// 		// 	// if(APS_num[4]>1)	printf("number of merging process = %d \n",APS_num[4]);

// 		// 	// cudaMemcpy(file_P1,dev_P1,num_part3*sizeof(part1),cudaMemcpyDeviceToHost);
// 		// 	// cudaMemcpy(file_P3,dev_P3,num_part3*sizeof(part3),cudaMemcpyDeviceToHost);
// 		// 	// save_vtk_bin_single(file_P1,file_P3);

// 		// 	t.x=1;
// 		// 	b.x=(num_part3-1)/t.x+1;
// 		// 	if(dim==2) KERNEL_particle_merging2D<<<b,t>>>(dev_P1,g_str,g_end,aps_num_part,dev_APS_num,aps,mutex);
// 		// 	// if(dim==3) KERNEL_particle_merging3D<<<b,t>>>(dev_P1,g_str,g_end,aps_num_part,dev_APS_num);
// 		// 	cudaDeviceSynchronize();

// 		// }
// 	num_part=k_num_part;
// 	cudaFree(mutex);
// 	}
// 	cudaFree(dev_APS_num);
// 	}

	//-------------------------------------------------------------------------------------------------
	// 출력
	//-------------------------------------------------------------------------------------------------

	if((time>=plotcount[0]*time_output)&&(time<(plotcount[0]+1)*time_output)){
		plotcount[0]+=1;

		int_t integer=ceil(time/time_output-0.5);

		printf("save plot...........................\n");
		cudaMemcpy(file_P1,dev_P1,num_part3*sizeof(part1),cudaMemcpyDeviceToHost);
		cudaMemcpy(file_P2,dev_SP2,num_part3*sizeof(part2),cudaMemcpyDeviceToHost);
		cudaMemcpy(file_P3,dev_P3,num_part3*sizeof(part3),cudaMemcpyDeviceToHost);

		save_vtk_bin_single(file_P1,file_P2,file_P3);
		
		// save_restart(file_P1,file_P2,file_P3);


		if(count==0) save_plot_fluid_vtk_bin_boundary(file_P1);
		pressureprobe(time,file_P1,file_P3,&time0[integer],&P0[integer],&P1[integer],&P2[integer],&P3[integer],time0,P0,P1,P2,P3);

		printf("time = %5.6f\n\n\n",time);
	 }
}

//-------------------------------------------------------------------------------------------------
// SOPHIA 메인 코드
//-------------------------------------------------------------------------------------------------
void*ISPH_Calc(void*arg){

	// 함수의 인자를 받아서 tid 에 저장 (tid = gpu 번호)
	int*idPtr,tid;
	idPtr=(int*)arg;
	tid=*idPtr;

	// timestep control 을 위한 변수 설정
	Real dt_CFL,V_MAX,K_stiff,eta;
	Real h0=HP1[0].h;

	dt_CFL=V_MAX=K_stiff=eta=0.0;
	num_cells=clc_num_cells();

	count=floor(time/dt+0.5);


	//-------------------------------------------------------------------------------------------------
	// Device 입자 생성
	//-------------------------------------------------------------------------------------------------

	// 계산할 GPU 정의: tid=GPU number
	cudaSetDevice(tid);

	// 출력할 변수 선언 및 메모리 할당
	part1*file_P1;
	file_P1=(part1*)malloc(sizeof(part1)*num_part3);
	memset(file_P1,0,sizeof(part1)*num_part3);

	part2*file_P2;
	part3*file_P3;

	file_P2=(part2*)malloc(sizeof(part2)*num_part3);
	memset(file_P2,0,sizeof(part2)*num_part3);

	file_P3=(part3*)malloc(sizeof(part3)*num_part3);
	memset(file_P3,0,sizeof(part3)*num_part3);

	//-------------------------------------------------------------------------------------------------
	// Device/GPU 변수 선언 및 메모리 할당
	//-------------------------------------------------------------------------------------------------

	// NNPS 관련 변수
	int_t*g_idx,*p_idx,*g_idx_in,*p_idx_in,*g_str,*g_end;

	// 주요 입자 변수
	part1*dev_P1,*dev_SP1;
	part2*dev_P2,*dev_SP2;
	part3*dev_SP3;

	// P2P 데이터 변수 선언
	int*p2p_af_in,*p2p_idx_in,*p2p_af,*p2p_idx;

	// // for Adaptive Particle Definement
	// int*aps_num_part;
	// cudaMalloc((void**)&aps_num_part,sizeof(int));
	// cudaMemset(aps_num_part,k_num_part,sizeof(int));

	// int*aps;
	// cudaMalloc((void**)&aps,sizeof(int));
	// cudaMemset(aps,0,sizeof(int));

	// NNPS 입자 메모리 할당
	cudaMalloc((void**)&g_idx,sizeof(int_t)*num_part3);
	cudaMalloc((void**)&p_idx,sizeof(int_t)*num_part3);
	cudaMalloc((void**)&g_idx_in,sizeof(int_t)*num_part3);
	cudaMalloc((void**)&p_idx_in,sizeof(int_t)*num_part3);
	cudaMalloc((void**)&g_str,sizeof(int_t)*num_cells);
	cudaMalloc((void**)&g_end,sizeof(int_t)*num_cells);

	// Device 입자 데이터 메모리 할당
	cudaMalloc((void**)&dev_P1,sizeof(part1)*num_part3);
	cudaMalloc((void**)&dev_SP1,sizeof(part1)*num_part3);
	cudaMalloc((void**)&dev_P2,sizeof(part2)*num_part3);
	cudaMalloc((void**)&dev_SP2,sizeof(part2)*num_part3);
	cudaMalloc((void**)&dev_SP3,sizeof(part3)*num_part3);

	// NNPS 메모리 초기화
	cudaMemset(g_idx_in,0,sizeof(int_t)*num_part3);
	cudaMemset(p_idx_in,0,sizeof(int_t)*num_part3);
	cudaMemset(g_idx,0,sizeof(int_t)*num_part3);
	cudaMemset(p_idx,0,sizeof(int_t)*num_part3);
	cudaMemset(g_str,cu_memset,sizeof(int_t)*num_cells);
	cudaMemset(g_end,0,sizeof(int_t)*num_cells);

	// Device 입자 메모리 초기화
	cudaMemset(dev_P1,0,sizeof(part1)*num_part3);
	cudaMemset(dev_SP1,0,sizeof(part1)*num_part3);
	cudaMemset(dev_P2,0,sizeof(part2)*num_part3);
	cudaMemset(dev_SP2,0,sizeof(part2)*num_part3);
	cudaMemset(dev_SP3,0,sizeof(part3)*num_part3);

	//-------------------------------------------------------------------------------------------------
	// Device/GPU로 데이터 복사
	//-------------------------------------------------------------------------------------------------

	// Sovler 전역변수 Device로 복사
	cudaMemcpyToSymbol(k_vii,vii,sizeof(int_t)*vii_size);
	cudaMemcpyToSymbol(k_vif,vif,sizeof(Real)*vif_size);

	// 물성 Table 데이터 Device로 복사
	initializePropertyTables();

	// Host 입자정보(HP1)를 분할하여(DHP1) Device로 복사(dev_P1)
	DHP1[tid]=(part1*)malloc(num_part3*sizeof(part1));
	memset(DHP1[tid],0,sizeof(part1)*num_part3);

	// 모든 입자를 더미로 초기화
	for(int i=0;i<num_part3;i++) DHP1[tid][i].i_type=3;

	c_initial_inner_outer_particle_single(HP1,DHP1[tid],tid);											// (CAUTION)
	cudaMemcpy(dev_P1,DHP1[tid],num_part3*sizeof(part1),cudaMemcpyHostToDevice);	// single gpu 이면 그냥 HP1을 device에 복사
	

	if(tid==0){
		printf("\n-----------------------------------------------------------\n");
		printf("GPU Domain Division Success\n");
		printf("-----------------------------------------------------------\n\n");
	}


	//-------------------------------------------------------------------------------------------------
// 화면 출력용 기타 변수들 정의 및 메모리 할당 (최대속도, 최대힘 등)
//-------------------------------------------------------------------------------------------------

// host
Real *max_umag0,*max_rho0,*max_ftotal0,*max_phi0;
max_umag0=(Real*)malloc(sizeof(Real));
max_rho0=(Real*)malloc(sizeof(Real));
max_ftotal0=(Real*)malloc(sizeof(Real));
max_phi0=(Real*)malloc(sizeof(Real));
max_umag0[0]=max_ftotal0[0]=max_rho0[0]=max_phi0[0]=0.0;

Real *dt10, *dt20, *dt30, *dt40, *dt50;
dt10=(Real*)malloc(sizeof(Real));
dt20=(Real*)malloc(sizeof(Real));
dt30=(Real*)malloc(sizeof(Real));
dt40=(Real*)malloc(sizeof(Real));
dt50=(Real*)malloc(sizeof(Real));
dt10[0]=dt20[0]=dt30[0]=dt40[0]=dt50[0]=0.0;

// device
Real*max_rho,*max_umag,*max_ft,*max_phi,*d_max_umag0,*d_max_rho0,*d_max_ftotal0,*d_max_phi0;
cudaMalloc((void**)&max_rho,sizeof(Real)*num_part3);
cudaMalloc((void**)&max_umag,sizeof(Real)*num_part3);
cudaMalloc((void**)&max_ft,sizeof(Real)*num_part3);
cudaMalloc((void**)&max_phi,sizeof(Real)*num_part3);
cudaMalloc((void**)&d_max_umag0,sizeof(Real));
cudaMalloc((void**)&d_max_rho0,sizeof(Real));
cudaMalloc((void**)&d_max_ftotal0,sizeof(Real));
cudaMalloc((void**)&d_max_phi0,sizeof(Real));
cudaMemset(max_umag,0,sizeof(Real)*num_part3);
cudaMemset(max_rho,0,sizeof(Real)*num_part3);
cudaMemset(max_ft,0,sizeof(Real)*num_part3);
cudaMemset(max_phi,0,sizeof(Real)*num_part3);
cudaMemset(d_max_umag0,0,sizeof(Real));
cudaMemset(d_max_rho0,0,sizeof(Real));
cudaMemset(d_max_ftotal0,0,sizeof(Real));
cudaMemset(d_max_phi0,0,sizeof(Real));

Real *dt1, *dt2, *dt3, *dt4, *dt5;
cudaMalloc((void**)&dt1,sizeof(Real)*num_part3);
cudaMalloc((void**)&dt2,sizeof(Real)*num_part3);
cudaMalloc((void**)&dt3,sizeof(Real)*num_part3);
cudaMalloc((void**)&dt4,sizeof(Real)*num_part3);
cudaMalloc((void**)&dt5,sizeof(Real)*num_part3);
cudaMemset(dt1,0,sizeof(Real)*num_part3);
cudaMemset(dt2,0,sizeof(Real)*num_part3);
cudaMemset(dt3,0,sizeof(Real)*num_part3);
cudaMemset(dt4,0,sizeof(Real)*num_part3);
cudaMemset(dt5,0,sizeof(Real)*num_part3);
Real *d_dt10, *d_dt20, *d_dt30, *d_dt40, *d_dt50;
cudaMalloc((void**)&d_dt10,sizeof(Real));
cudaMalloc((void**)&d_dt20,sizeof(Real));
cudaMalloc((void**)&d_dt30,sizeof(Real));
cudaMalloc((void**)&d_dt40,sizeof(Real));
cudaMalloc((void**)&d_dt50,sizeof(Real));
	//-------------------------------------------------------------------------------------------------
	// 정렬(Sorting)을 위한 CUB 라이브러리 변수 준비
	//-------------------------------------------------------------------------------------------------

	// Sorting & Max variable to use CUB Library
	void*dev_sort_storage=NULL;
	void*dev_max_storage=NULL;
	size_t sort_storage_bytes=0;
	size_t max_storage_bytes=0;

	// Determine Sorting & Maximum Value Setting for Total Particle Data
	cub::DeviceRadixSort::SortPairs(dev_sort_storage,sort_storage_bytes,g_idx_in,g_idx,p_idx_in,p_idx,num_part3);
	cub::DeviceReduce::Max(dev_max_storage,max_storage_bytes,max_umag,d_max_umag0,num_part3);
	cudaDeviceSynchronize();
	cudaMalloc((void**)&dev_sort_storage,sort_storage_bytes);
	cudaMalloc((void**)&dev_max_storage,max_storage_bytes);
	

	//-------------------------------------------------------------------------------------------------
	// 코드 메인
	//-------------------------------------------------------------------------------------------------

	// 초기상태 및 설정 출력
	if(tid==0){
		printf("-----------------------------------------------------------\n");
		printf("Input Summary: \n");
		printf("-----------------------------------------------------------\n");
		printf("	Total number of particles=%d\n",num_part);
		printf("	Device number of particles=%d\n",num_part3);
		printf("	P2P number of particles=%d\n",num_p2p);
		printf("	NI=%d,	NJ=%d,	NK=%d\n",NI,NJ,NK);
		printf("-----------------------------------------------------------\n\n");
		// Input Check
		printf("-----------------------------------------------------------\n");
		printf("Input Check: \n");
		printf("-----------------------------------------------------------\n");
		// check Domain Status
		printf("x min, max : %f %f\n",x_min,x_max);
		printf("y min, max : %f %f\n",y_min,y_max);
		printf("z min, max : %f %f\n",z_min,z_max);
		printf("Cell Size(dcell) %f\n",dcell);
		printf("Number of Cells Per a GPU in x-direction(calc_area) %d\n",calc_area);
		printf("-----------------------------------------------------------\n\n");
		// print out status
		printf("\n");
		printf("-----------------------------\n");
		printf("Start Simultion!!\n");
		printf("-----------------------------\n");
		printf("\n");
	}
	

	//-------------------------------------------------------------------------------------------------
	// 코드 메인
	//-------------------------------------------------------------------------------------------------

	int_t N = 2*time_end/time_output;
	Real Cdp[N], Cdv[N], Cd[N];
	Real Clp[N], Clv[N], Cl[N];
	int_t plotcount[1];
	plotcount[0]=0;
	Real P0[N], P1[N], P2[N], P3[N];
	Real time0[N];
	for(int i=0; i<N; i++)
	{
		Cdp[i]=0;
		Cdv[i]=0;
		Cd[i]=0;
		Clp[i]=0;
		Clv[i]=0;
		Cl[i]=0;
		P0[i]=0.0;
		P1[i]=0.0;
		P2[i]=0.0;
		P3[i]=0.0;
		time0[i]=0.0;
	}
	plotcount[0]=0;

		clock_t start=clock();

		while(1){
		
			//-------------------------------------------------------------------------------------------------
			// Time-step Control
			//-------------------------------------------------------------------------------------------------
			if(tid==0){
				//timestep is updated every 10 steps ------------ estimate new timestep (Goswami & Pajarola(2011))
	
	
					dim3 b,t;
					t.x=128;
					b.x=(num_part3-1)/t.x+1;
					kernel_copy_max_timestep<<<b,t>>>(dev_P1,dev_SP2,dev_SP3,dt1, dt2, dt3, dt4, dt5);
					cudaDeviceSynchronize();
					cudaError_t err = cudaGetLastError();
					if (err != cudaSuccess) {
						printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
					}

					// Find Max Velocity & Force using CUB - TID=0
					cub::DeviceReduce::Max(dev_max_storage,max_storage_bytes,max_phi,d_max_phi0,num_part3);
					cub::DeviceReduce::Max(dev_max_storage,max_storage_bytes,max_ft,d_max_ftotal0,num_part3);
					cub::DeviceReduce::Max(dev_max_storage,max_storage_bytes,max_umag,d_max_umag0,num_part3);
					cudaDeviceSynchronize();
					cudaMemcpy(max_phi0,d_max_phi0,sizeof(Real),cudaMemcpyDeviceToHost);
					cudaMemcpy(max_ftotal0,d_max_ftotal0,sizeof(Real),cudaMemcpyDeviceToHost);
					cudaMemcpy(max_umag0,d_max_umag0,sizeof(Real),cudaMemcpyDeviceToHost);
	
					cub::DeviceReduce::Max(dev_max_storage,max_storage_bytes,dt1,d_dt10,num_part3);
					cudaDeviceSynchronize();
					cudaMemcpy(dt10,d_dt10,sizeof(Real),cudaMemcpyDeviceToHost);
					cub::DeviceReduce::Max(dev_max_storage,max_storage_bytes,dt2,d_dt20,num_part3);
					cudaDeviceSynchronize();
					cudaMemcpy(dt20,d_dt20,sizeof(Real),cudaMemcpyDeviceToHost);
					cub::DeviceReduce::Max(dev_max_storage,max_storage_bytes,dt3,d_dt30,num_part3);
					cudaDeviceSynchronize();
					cudaMemcpy(dt30,d_dt30,sizeof(Real),cudaMemcpyDeviceToHost);
					cub::DeviceReduce::Max(dev_max_storage,max_storage_bytes,dt4,d_dt40,num_part3);
					cudaDeviceSynchronize();
					cudaMemcpy(dt40,d_dt40,sizeof(Real),cudaMemcpyDeviceToHost);
					cub::DeviceReduce::Max(dev_max_storage,max_storage_bytes,dt5,d_dt50,num_part3);
					cudaDeviceSynchronize();
					cudaMemcpy(dt50,d_dt50,sizeof(Real),cudaMemcpyDeviceToHost);
	
					// Real h0 = HP1[0].h;
					// Real dt_delta = 0.44*h0/delta/soundspeed;
					// Real dt_vel = 0.25*(h0/(soundspeed))/sqrt(1000);
					// Real dt_ft = 0.25*sqrt(h0/(max_ftotal0[0]+1E-10));
					// Real dt_vis = 0.25*(h0*h0/0.01);
					Real check_dt;
					cudaMemcpy(&check_dt, d_dt10, sizeof(Real), cudaMemcpyDeviceToHost);
					printf("🔍 dt10 = %e\n", check_dt);  // 0이면 문제

					Real dt_delta = 0.1;
					Real dt_vel = 1.0/dt10[0];
					Real dt_ft = 0.1;
					Real dt_vis = 1.0/dt20[0];
					Real timestep[4] = {dt_delta,dt_vel,dt_ft,dt_vis};
					dt = min(timestep[0],timestep[1]);
					dt = min(dt,timestep[2]);
					dt = min(dt,timestep[3]);
	
					int_t integer=((time>=plotcount[0]*time_output)&&(time<(plotcount[0]+1)*time_output));
	
					if(integer) printf("dt_delta = %2.2e\ndt_vel = %2.2e\ndt_ft = %2.2e\ndt_vis = %2.2e\n",timestep[0],timestep[1],timestep[2],timestep[3]);
					if(integer) printf("dt = %e[s]\n",dt);
	
					if(ngpu==1){
						SOPHIA_single_ISPH(g_idx,p_idx,g_idx_in,p_idx_in,g_str,g_end,dev_P1,dev_SP1,dev_P2,dev_SP2,dev_SP3,
							p2p_af_in,p2p_idx_in,p2p_af,p2p_idx,dev_sort_storage,&sort_storage_bytes,file_P1,file_P2,file_P3,tid
							,time0,P0,P1,P2,P3,plotcount);
						}
	
					if(integer){
						kernel_copy_max<<<b,t>>>(dev_P1,dev_SP2,dev_SP3,max_rho,max_ft,max_umag);
						cudaDeviceSynchronize();
					// Find Max Velocity & Force using CUB - TID=0
					cub::DeviceReduce::Max(dev_max_storage,max_storage_bytes,max_umag,d_max_umag0,num_part3);
					cub::DeviceReduce::Max(dev_max_storage,max_storage_bytes,max_rho,d_max_rho0,num_part3);
					cub::DeviceReduce::Max(dev_max_storage,max_storage_bytes,max_ft,d_max_ftotal0,num_part3);
					cudaDeviceSynchronize();
					cudaMemcpy(max_umag0,d_max_umag0,sizeof(Real),cudaMemcpyDeviceToHost);
					cudaMemcpy(max_rho0,d_max_rho0,sizeof(Real),cudaMemcpyDeviceToHost);
					cudaMemcpy(max_ftotal0,d_max_ftotal0,sizeof(Real),cudaMemcpyDeviceToHost);
					printf("dt = %e[s]\n",dt);
					printf("%d\t compress_max=%5.2f%\tu_max=%5.2f\tftotal_max=%5.2f\n\n",count,max_rho0[0],max_umag0[0],max_ftotal0[0]);
				}
				time+=dt;
				count++;
			}
	
			
			if(time>=time_end) break;
	
		}
	
		clock_t end=clock();
		clock_t elapsed = end-start;
		double calctime=(double)(elapsed/CLOCKS_PER_SEC);
		printf("total calculation time = %f\n",calctime);

	//-------------------------------------------------------------------------------------------------
	// ##. Save Restart File
	//-------------------------------------------------------------------------------------------------

	if(ngpu==1) {

		save_restart(file_P1,file_P2,file_P3);
		cudaMemcpy(file_P1,dev_SP1,num_part3*sizeof(part1),cudaMemcpyDeviceToHost);
		cudaMemcpy(file_P2,dev_SP2,num_part3*sizeof(part2),cudaMemcpyDeviceToHost);
		cudaMemcpy(file_P3,dev_SP3,num_part3*sizeof(part3),cudaMemcpyDeviceToHost);


		free(file_P2);
		free(file_P3);
	}

	//-------------------------------------------------------------------------------------------------
	// ##. Memory Free
	//-------------------------------------------------------------------------------------------------
	free(file_P1);
	free(max_umag0);
	free(max_rho0);
	free(max_ftotal0);
	cudaFree(g_idx);
	cudaFree(p_idx);
	cudaFree(g_idx_in);
	cudaFree(p_idx_in);
	cudaFree(g_str);
	cudaFree(g_end);
	cudaFree(dev_P1);
	cudaFree(dev_SP1);
	cudaFree(dev_P2);
	cudaFree(dev_SP2);
	cudaFree(dev_SP3);
	cudaFree(max_umag);
	cudaFree(max_rho);
	cudaFree(max_ft);
	cudaFree(dt1);
	cudaFree(dt2);
	cudaFree(dt3);
	cudaFree(dt4);
	cudaFree(dt5);
	cudaFree(d_dt10);
	cudaFree(d_dt20);
	cudaFree(d_dt30);
	cudaFree(d_dt40);
	cudaFree(d_dt50);
	cudaFree(d_max_umag0);
	cudaFree(d_max_rho0);
	cudaFree(d_max_ftotal0);
	cudaFree(dev_sort_storage);
	cudaFree(dev_max_storage);
	

	return 0;
}
