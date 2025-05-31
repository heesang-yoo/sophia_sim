
//////////////////////////////////////////////////////////////////////////////
#define L1 	-0.0625				// min(y)
#define L2	0.0		// min(y)+3.0*initial_particle_spacing
#define L3	1.0		// max(y)-3.0*initial_particle_spacing
#define L4	1.0625			// max(y)
#define L5	-0.1

#define L_left 0.00 		// min(x)+3.0*initial_particle_spacing
#define L_right 0.04	// max(x)-3.0*initial_particle_spacing
//////////////////////////////////////////////////////////////////////////////

void c_initial_inner_outer_particle_single(part1*HP1,part1*DHP1,int_t tid){

	int_t i,c_count;
	Real xi0;
	Real maxb,minb;

	c_count=0;

	for(i=0;i<num_part;i++){
		HP1[i].i_type=1;
		HP1[i].buffer_type=0;

		if (open_boundary>0)			// (CAUTION)
		{
			if(HP1[i].p_type>=1){
				if(HP1[i].y>=(L1-1e-6)&&(HP1[i].y<L2-1e-6)) {
					HP1[i].buffer_type=1;
					HP1[i].i_type=2;
				}
				else if(HP1[i].y>=(L2-1e-6)&&(HP1[i].y<L3)) {
					HP1[i].buffer_type=0;
					HP1[i].i_type=1;
				}
				else if ((HP1[i].y>(L3+1e-6))&&(HP1[i].y<=(1.001*L4))) {
					HP1[i].buffer_type=2;
					HP1[i].i_type=2;
				}
			}

			// if(HP1[i].p_type<=0){
			// 	if(HP1[i].x<1e-6+L_left) {
			// 		HP1[i].buffer_type=3;
			// 		HP1[i].i_type=2;
			// 	}
			// 	else if(HP1[i].x>L_right-1e-6) {
			// 		HP1[i].buffer_type=4;
			// 		HP1[i].i_type=2;
			// 	}
			// }
		}

		DHP1[c_count]=HP1[i];
		c_count++;
	}

}
//////////////////////////////////////////////////////////////////////////////
void c_initial_inner_outer_particle(part1*HP1,part1*DHP1,int_t tid){

	int_t i,c_count;
	Real xi0;
	Real maxb,minb;

	// 각 GPU의 x 축 왼쪽 좌표(minb)와 오른쪽 좌표(maxb) 계산
	minb=x_min+(Real)(calc_area*tid)*dcell;
	maxb=x_min+(Real)(calc_area*(tid+1))*dcell;

	// 입자의 개수(혹은 위치) 변수
	int_t cpucount;
	cpucount=0;

	// Host 입자정보(HP1)를 GPU별로 분할(DHP1)
	if(tid==0){		// 첫번째 GPU 라면
		c_count=0;
		for(i=0;i<num_part;i++){
			xi0=HP1[i].x;

			if(xi0<maxb&&xi0>=x_min) cpucount++;	// 입자의 위치가 영역안에 들어오는지 판별

			if(xi0>=maxb+dcell*2.0||xi0<x_min) continue;	// 입자의 위치가 계산영역(버퍼 포함)을 벗어나는지 판별
			if(xi0>=maxb&&xi0<maxb+dcell*2.0){	// 입자의 위치가 outer buffer 영역에 포함되는지 판별
				HP1[i].i_type=2;		// 버퍼영역이면 i_type=2
				DHP1[c_count]=HP1[i];	// 정보 복사
				c_count++;
				continue;
			}
			if((xi0<(maxb-dcell*2.0))&&(xi0>=(x_min+dcell*2.0))){		// 입자의 위치가 inner 영역에 있는지 판별
				HP1[i].i_type=0;
				DHP1[c_count]=HP1[i];
				c_count++;
			}else{		// 그 외에는 inner buffer 영역으로 간주
				HP1[i].i_type=1;
				DHP1[c_count]=HP1[i];
				c_count++;
			}
		}
	}else if(tid==ngpu-1){		// 마지막 GPU 라면
		c_count=0;
		for(i=0;i<num_part;i++){
			xi0=HP1[i].x;

			if(xi0<=x_max&&xi0>=minb) cpucount++;

			if(xi0>x_max||xi0<minb-dcell*2.0) continue;
			if(xi0<minb&&xi0>=minb-dcell*2.0){
				HP1[i].i_type=2;
				DHP1[c_count]=HP1[i];
				c_count++;
				continue;
			}
			if((xi0<(x_max-dcell*2.0))&&(xi0>=(minb+dcell*2.0))){
				HP1[i].i_type=0;
				DHP1[c_count]=HP1[i];
				c_count++;
			}else{
				HP1[i].i_type=1;
				DHP1[c_count]=HP1[i];
				c_count++;
			}
		}
	}else{	// 그 밖의 GPU에 대해서
		c_count=0;
		for(i=0;i<num_part;i++){
			xi0=HP1[i].x;

			if(xi0<maxb&&xi0>=minb) cpucount++;

			if(xi0>=maxb+dcell*2.0||xi0<minb-dcell*2.0) continue;
			if(xi0>=maxb&&xi0<maxb+dcell*2.0){
				HP1[i].i_type=2;
				DHP1[c_count]=HP1[i];
				c_count++;
				continue;
			}
			if(xi0<minb&&xi0>=minb-dcell*2.0){
				HP1[i].i_type=2;
				DHP1[c_count]=HP1[i];
				c_count++;
				continue;
			}
			if((xi0<(maxb-dcell*2.0))&&(xi0>=(minb+dcell*2.0))){
				HP1[i].i_type=0;
				DHP1[c_count]=HP1[i];
				c_count++;
			}else{
				HP1[i].i_type=1;
				DHP1[c_count]=HP1[i];
				c_count++;
			}
		}
	}

	// if inner else outer
	//if((xi0<=(maxb-dcell*1.0))&&(xi0>=(minb+dcell*1.0))) HP1[i].i_type=1;
	//else HP1[i].i_type=2;
	printf("%d Particle Number : %d\n",tid,cpucount);
	printf("%d Ref+Real Number : %d\n",tid,c_count);

}
////////////////////////////////////////////////////////////////////////
// find morton 2d curve index (for z-idx)
__host__ __device__ uint64_t morton2d(uint64_t x,uint64_t y)
{
	uint64_t z=0;

	x=(x|(x<<16))&0x0000FFFF0000FFFF;
	x=(x|(x<<8))&0x00FF00FF00FF00FF;
	x=(x|(x<<4))&0x0F0F0F0F0F0F0F0F;
	x=(x|(x<<2))&0x3333333333333333;
	x=(x|(x<<1))&0x5555555555555555;
	y=(y|(y<<16))&0x0000FFFF0000FFFF;
	y=(y|(y<<8))&0x00FF00FF00FF00FF;
	y=(y|(y<<4))&0x0F0F0F0F0F0F0F0F;
	y=(y|(y<<2))&0x3333333333333333;
	y=(y|(y<<1))&0x5555555555555555;

	z=x|(y<<1);

	return z;
}
////////////////////////////////////////////////////////////////////////
// find morton 3d curve index (for z-idx)
__host__ __device__ uint64_t morton3d(unsigned int a,unsigned int b,unsigned int c)
{
	uint64_t answer=0;

	uint64_t x=a&0x1fffff;// we only look at the first 21 bits
	x=(x|x<<32)&0x1f00000000ffff; // shift left 32 bits,OR with self,and 00011111000000000000000000000000000000001111111111111111
	x=(x|x<<16)&0x1f0000ff0000ff; // shift left 32 bits,OR with self,and 00011111000000000000000011111111000000000000000011111111
	x=(x|x<<8)&0x100f00f00f00f00f;// shift left 32 bits,OR with self,and 0001000000001111000000001111000000001111000000001111000000000000
	x=(x|x<<4)&0x10c30c30c30c30c3;// shift left 32 bits,OR with self,and 0001000011000011000011000011000011000011000011000011000100000000
	x=(x|x<<2)&0x1249249249249249;

	uint64_t y=b&0x1fffff;// we only look at the first 21 bits
	y=(y|y<<32)&0x1f00000000ffff; // shift left 32 bits,OR with self,and 00011111000000000000000000000000000000001111111111111111
	y=(y|y<<16)&0x1f0000ff0000ff; // shift left 32 bits,OR with self,and 00011111000000000000000011111111000000000000000011111111
	y=(y|y<<8)&0x100f00f00f00f00f;// shift left 32 bits,OR with self,and 0001000000001111000000001111000000001111000000001111000000000000
	y=(y|y<<4)&0x10c30c30c30c30c3;// shift left 32 bits,OR with self,and 0001000011000011000011000011000011000011000011000011000100000000
	y=(y|y<<2)&0x1249249249249249;

	uint64_t z=c&0x1fffff;// we only look at the first 21 bits
	z=(z|z<<32)&0x1f00000000ffff; // shift left 32 bits,OR with self,and 00011111000000000000000000000000000000001111111111111111
	z=(z|z<<16)&0x1f0000ff0000ff; // shift left 32 bits,OR with self,and 00011111000000000000000011111111000000000000000011111111
	z=(z|z<<8)&0x100f00f00f00f00f;// shift left 32 bits,OR with self,and 0001000000001111000000001111000000001111000000001111000000000000
	z=(z|z<<4)&0x10c30c30c30c30c3;// shift left 32 bits,OR with self,and 0001000011000011000011000011000011000011000011000011000100000000
	z=(z|z<<2)&0x1249249249249249;

	answer|=x|y<<1|z<<2;
	return answer;
}
//////////////////////////////////////////////////////////////////////////////
// 필요한 cell의 개수
int_t clc_num_cells() {

	int_t result;
	int_t NI_max;

	NI_max=max(max(NI,NJ),NK);

	//printf("NI_max = %d\n", NI_max);

	if (flag_z_index==0) {
		result=NI*NJ*NK+1;
	}
	else {
		if (dim==2) result=morton2d(NI_max-1,NI_max-1);
		if (dim==3) result=morton3d(NI_max-1,NI_max-1,NI_max-1);
	}

	return result;
}
//////////////////////////////////////////////////////////////////////////////
// cell의 index
__host__ __device__ int_t idx_cell(int_t Ix, int_t Iy, int_t Iz) {

	int_t result;

	if (k_flag_z_index==0) {
		result=(Ix)+k_NI*(Iy)+k_NI*k_NJ*(Iz);
	}
	else {
		if (k_dim==2) result=morton2d(Ix,Iy);
		if (k_dim==3) result=morton3d(Ix,Iy,Iz);
	}
	return result;
}
//////////////////////////////////////////////////////////////////////////////
__global__ void right_send_particle(int_t*p2p_af,int_t*p2p_idx,part1*P1,part3*P3,part1*SP1,p2p_part3*SP3,int_t tid){
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part3) return;
	if(P1[i].i_type!=1){		// 입자가 처음부터 영역 1 에 있지 않았으면 (영역 0 또는 영역 2, 3)
		p2p_af[i]=2;					// 표식의 우선순위를 낮추고
		p2p_idx[i]=i;
		SP1[i].i_type=3;			// 더미 입자로 설정
		return;
	}

	Real maxb,xi0;
	xi0=P1[i].x;

	maxb=k_x_min+k_calc_area*k_dcell*(tid+1);

	if(xi0>=maxb-k_dcell*2.0){
		p2p_af[i]=1;
		p2p_idx[i]=i;

		SP1[i]=P1[i];
		SP3[i].drho=P3[i].drho;
		SP3[i].dconcn=P3[i].dconcn;
		SP3[i].denthalpy=P3[i].denthalpy;
		SP3[i].ftotalx=P3[i].ftotalx;
		SP3[i].ftotaly=P3[i].ftotaly;
		SP3[i].ftotalz=P3[i].ftotalz;
		SP3[i].ftotal=P3[i].ftotal;

		if(xi0>=maxb) SP1[i].i_type=1;
		else SP1[i].i_type=2;
	}else{
		p2p_af[i]=2;
		p2p_idx[i]=i;
		SP1[i].i_type=3;
	}
}
//////////////////////////////////////////////////////////////////////////////
__global__ void left_send_particle(int_t*p2p_af,int_t*p2p_idx,part1*P1,part3*P3,part1*SP1,p2p_part3*SP3,int_t tid){
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part3) return;
	if(P1[i].i_type!=1){
		p2p_af[i]=2;
		p2p_idx[i]=i;
		SP1[i].i_type=3;
		return;
	}

	Real minb,xi0;
	xi0=P1[i].x;

	minb=k_x_min+k_calc_area*k_dcell*tid;

	if(xi0<minb+k_dcell*2.0){
		p2p_af[i]=1;
		p2p_idx[i]=i;

		SP1[i]=P1[i];
		SP3[i].drho=P3[i].drho;
		SP3[i].dconcn=P3[i].dconcn;
		SP3[i].denthalpy=P3[i].denthalpy;
		SP3[i].ftotalx=P3[i].ftotalx;
		SP3[i].ftotaly=P3[i].ftotaly;
		SP3[i].ftotalz=P3[i].ftotalz;
		SP3[i].ftotal=P3[i].ftotal;

		if(xi0<minb) SP1[i].i_type=1;
		else SP1[i].i_type=2;

	}else{
		p2p_af[i]=2;
		p2p_idx[i]=i;
		SP1[i].i_type=3;
	}
}
//////////////////////////////////////////////////////////////////////////////
__global__ void reorder_data_p2p(int_t*p2p_af,int_t*p2p_idx,part1*SP1,part1*s_SP1,p2p_part3*SP3,p2p_part3*s_SP3){
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_p2p) return;
	//if(p2p_af[i]==2) return;

	int sortedIndex=p2p_idx[i];

	s_SP1[i]=SP1[sortedIndex];
	s_SP3[i]=SP3[sortedIndex];
}
//////////////////////////////////////////////////////////////////////////////
__global__ void p2p_copyData(part1*SP1,part3*SP3,part1*rP1,p2p_part3*rP3,int_t tid){
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_p2p) return;
	//if(rP1[i].i_type!=1) return;

	int ps=k_num_part3-i-1;
	int ri=rP1[i].i_type;

	// if(SP1[ps].i_type==1) return;
	if(ri==3){
		SP1[ps].i_type=3;		// 들어온 입자가 더미이면 더미로 기록
	}else{								// 아니면 정보 복사
		SP1[ps]=rP1[i];
		SP3[ps].drho=rP3[i].drho;
		SP3[ps].dconcn=rP3[i].dconcn;
		SP3[ps].denthalpy=rP3[i].denthalpy;
		SP3[ps].ftotalx=rP3[i].ftotalx;
		SP3[ps].ftotaly=rP3[i].ftotaly;
		SP3[ps].ftotalz=rP3[i].ftotalz;
		SP3[ps].ftotal=rP3[i].ftotal;
	}
	rP1[i].i_type=3;			// 작업이 끝나면 더미로 설정
}
//////////////////////////////////////////////////////////////////////////////
__global__ void init_Recv(part1*rP1){
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_p2p) return;

	rP1[i].i_type=3;
}
//////////////////////////////////////////////////////////////////////////////
__global__ void initial_particle(part1*P1,int_t tid){
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part3) return;

	if(P1[i].i_type==3) return;
	if(P1[i].i_type==2){
		P1[i].i_type=3;
		return;
	}

	Real xi0=P1[i].x;

	Real maxb,minb;

	minb=k_x_min+(Real)(k_calc_area*tid)*k_dcell;
	maxb=k_x_min+(Real)(k_calc_area*(tid+1))*k_dcell;

	if(tid==0){
		if(xi0>=maxb+k_dcell*2.0||xi0<k_x_min){
			P1[i].i_type=3;
			return;
		}
		if(xi0>=maxb&&xi0<maxb+k_dcell*2.0){
			P1[i].i_type=2;
			return;
		}
		if((xi0<(maxb-k_dcell*2.0))&&(xi0>=(k_x_min+k_dcell*2.0))){
			P1[i].i_type=0;
		}else{
			P1[i].i_type=1;
		}
	}else if(tid==k_ngpu-1){
		if(xi0>k_x_max||xi0<minb-k_dcell*2.0){
			P1[i].i_type=3;
			return;
		}
		if(xi0<minb&&xi0>=minb-k_dcell*2.0){
			P1[i].i_type=2;
			return;
		}
		if((xi0<(k_x_max-k_dcell*2.0))&&(xi0>=(minb+k_dcell*2.0))){
			P1[i].i_type=0;
		}else{
			P1[i].i_type=1;
		}
	}else{
		if(xi0>=maxb+k_dcell*2.0||xi0<minb-k_dcell*2.0){
			P1[i].i_type=3;
			return;
		}
		if(xi0>=maxb&&xi0<maxb+k_dcell*2.0){
			P1[i].i_type=2;
			return;
		}
		if(xi0<minb&&xi0>=minb-k_dcell*2.0){
			P1[i].i_type=2;
			return;
		}
		if((xi0<(maxb-k_dcell*2.0))&&(xi0>=(minb+k_dcell*2.0))){
			P1[i].i_type=0;
		}else{
			P1[i].i_type=1;
		}
	}

}
//////////////////////////////////////////////////////////////////////////////
__global__ void inner_outer_particle(part1*P1,int_t tid){
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part3) return;
	// if(P1[i].i_type==2||P1[i].i_type==3) return;
	if(P1[i].i_type==3) return;

	Real xi0=P1[i].x;

	Real maxb,minb;

	minb=k_x_min+(Real)(k_calc_area*tid)*k_dcell;
	maxb=k_x_min+(Real)(k_calc_area*(tid+1))*k_dcell;

	if(tid==0){
		if(xi0>=maxb+k_dcell*2.0||xi0<k_x_min){
			P1[i].i_type=3;
			return;
		}
		if(xi0>=maxb&&xi0<maxb+k_dcell*2.0){
			P1[i].i_type=2;
			return;
		}
		if((xi0<(maxb-k_dcell*2.0))&&(xi0>=(k_x_min+k_dcell*2.0))){
			P1[i].i_type=0;
		}else{
			P1[i].i_type=1;
		}
	}else if(tid==k_ngpu-1){
		if(xi0>k_x_max||xi0<minb-k_dcell*2.0){
			P1[i].i_type=3;
			return;
		}
		if(xi0<minb&&xi0>=minb-k_dcell*2.0){
			P1[i].i_type=2;
			return;
		}
		if((xi0<(k_x_max-k_dcell*2.0))&&(xi0>=(minb+k_dcell*2.0))){
			P1[i].i_type=0;
		}else{
			P1[i].i_type=1;
		}
	}else{
		if(xi0>=maxb+k_dcell*2.0||xi0<minb-k_dcell*2.0){
			P1[i].i_type=3;
			return;
		}
		if(xi0>=maxb&&xi0<maxb+k_dcell*2.0){
			P1[i].i_type=2;
			return;
		}
		if(xi0<minb&&xi0>=minb-k_dcell*2.0){
			P1[i].i_type=2;
			return;
		}
		if((xi0<(maxb-k_dcell*2.0))&&(xi0>=(minb+k_dcell*2.0))){
			P1[i].i_type=0;
		}else{
			P1[i].i_type=1;
		}
	}

}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_reorder(int_t*g_idx,int_t*p_idx,int_t*g_str,int_t*g_end,part1*P1,part2*P2,part1*SP1,part2*SP2)
{
	extern __shared__ int sharedHash[];

	int idx=threadIdx.x+blockIdx.x*blockDim.x;
	if(idx>=k_num_part3) return;
	int hash;

	hash=g_idx[idx];

	//if (hash>=k_num_cells && P1[idx].i_type!=3) printf("itype=%d hash %d k_num_cells %d\n", P1[idx].i_type, hash, k_num_cells);

	sharedHash[threadIdx.x+1]=hash;
	if(idx>0&&threadIdx.x==0){
		/*save the end of the previous block g_idx*/
		sharedHash[0]=g_idx[idx-1];
	}
	__syncthreads();		// for sorting and reorder particle property
	if(idx==0||hash!=sharedHash[threadIdx.x]){
		//if(hash<=k_num_cells) {
			g_str[hash]=idx;
			if(idx>0) g_end[sharedHash[threadIdx.x]]=idx;
		//}
	}
	// if((idx==k_num_part3-1)&&(hash<k_num_cells)) g_end[hash]=idx+1;
	if((idx==k_num_part3-1)) g_end[hash]=idx+1;
	/*reorder data*/
	int sortedIndex=p_idx[idx];

	SP1[idx]=P1[sortedIndex];
	SP2[idx]=P2[sortedIndex];
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_index_particle_to_cell(int_t*g_idx,int_t*p_idx,part1*P1)
{
	int_t idx=threadIdx.x+blockIdx.x*blockDim.x;
	if(idx>=k_num_part3) return;
	// Not in Each GPUs Domain
	if(P1[idx].i_type==3){
		g_idx[idx]=k_num_cells;
		p_idx[idx]=idx;
	}
	else {
		int_t icell,jcell,kcell;
		// calculate I,J,K in cell
		if((k_x_max==k_x_min)){icell=0;}
		else{icell=min(floor((P1[idx].x-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
		if((k_y_max==k_y_min)){jcell=0;}
		else{jcell=min(floor((P1[idx].y-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
		if((k_z_max==k_z_min)){kcell=0;}
		else{kcell=min(floor((P1[idx].z-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
		// out-of-range handling
		if(icell<0) icell=0;
		if(jcell<0) jcell=0;
		if(kcell<0) kcell=0;
		// calculate cell index from I,J,K
		p_idx[idx]=idx;
		g_idx[idx]=idx_cell(icell,jcell,kcell);


	}

	if(g_idx[idx]==k_num_cells && P1[idx].i_type!=3) printf("itype %d \n", P1[idx].i_type);
}
/////////////////////////////////////////////////////////////////////////
// Neighbor-Numbering and Particle Sorting (NNPS)
void NNPS(
    int_t* g_idx_in, int_t* g_idx,
    int_t* p_idx_in, int_t* p_idx,
    int_t* g_str, int_t* g_end,
    part1* dev_P1, part1* dev_SP1,
    part2* dev_P2, part2* dev_SP2,
    void* dev_sort_storage, size_t* sort_storage_bytes,
    dim3 b, dim3 t, int_t s) {

    // If this is the first step (Eulerian) or a Lagrangian update:
    if (((scheme == Eulerian) && (count == 0)) || (scheme > Eulerian)) {
        // 1. Reset g_str (cell start index)
        cudaMemset(g_str, cu_memset, sizeof(int_t) * num_cells);

        // 2. Assign cell indices to particles
        b.x = (num_part3 - 1) / t.x + 1;
        KERNEL_index_particle_to_cell<<<b, t>>>(g_idx_in, p_idx_in, dev_P1);
        cudaDeviceSynchronize();

        // 3. Sort particle indices by cell using CUB
        cub::DeviceRadixSort::SortPairs(
            dev_sort_storage, *sort_storage_bytes,
            g_idx_in, g_idx, p_idx_in, p_idx, num_part3
        );
        cudaDeviceSynchronize();

        // 4. Reorder particles based on sorted indices
        b.x = (num_part3 - 1) / t.x + 1;
        KERNEL_reorder<<<b, t, s>>>(g_idx, p_idx, g_str, g_end, dev_P1, dev_P2, dev_SP1, dev_SP2);
        cudaDeviceSynchronize();

        // (Optional) Reset dev_P3 if needed
        // cudaMemset(dev_P3, 0, sizeof(part3) * num_part3);

        // 5. Copy reordered particle data back to dev_P1
        cudaMemcpy(dev_P1, dev_SP1, sizeof(part1) * num_part3, cudaMemcpyDeviceToDevice);

    } else if ((scheme == Eulerian) && (count > 0)) {
        // For subsequent Eulerian steps (no sorting needed):

        // (Optional) Reset dev_P3 if needed
        // cudaMemset(dev_P3, 0, sizeof(part3) * num_part3);

        // 1. Copy dev_P1 to dev_SP1
        cudaMemcpy(dev_SP1, dev_P1, sizeof(part1) * num_part3, cudaMemcpyDeviceToDevice);

        // 2. Copy dev_P2 to dev_SP2
        cudaMemcpy(dev_SP2, dev_P2, sizeof(part2) * num_part3, cudaMemcpyDeviceToDevice);
    }
}
