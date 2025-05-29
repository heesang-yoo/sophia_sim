////////////////////////////////////////////////////////////////////////
#define NU0_HB		1.0			//Herschel-Bulkey model parameter
#define TAU0_HB		18.24		//Herschel-Bulkey model parameter (for lava flow)
#define K0_HB		1.90		//Herschel-Bulkey model parameter (for lava flow)
#define	N0_HB		0.53		//Herschel-Bulkey model parameter (for lava flow)
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_EOS(part1*P1,part2*P2)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>i_type_crt) return;
	if(k_solver_type==Icsph&&P1[i].p_type==1) return;
	if(P1[i].p_type<=0) return;

	Real tB,tP,rho0,tmpres;
	Real cc = k_soundspeed;

	rho0=fmax(1e-6,k_rho0_eos);
	tB=cc*cc*rho0/k_gamma;
	tP=pow(P1[i].rho/P2[i].rho_ref,k_gamma);

	//P1[i].pres=tB*(tP-1.0);
	tmpres=tB*(tP-1.0);
	P1[i].pres=tmpres;
}
////////////////////////////////////////////////////////////////////////
__host__ __device__ Real interp2(Real *x_data,Real *y_data,int size,Real x)
{

	Real y;
	int i;
	int end_idx=size-1;

	if(x_data[end_idx]<x){
		//y=y_data[end_idx]+(y_data[end_idx]-y_data[end_idx-1])/(x_data[end_idx]-x_data[end_idx-1])*(x-x_data[end_idx]);
		y=y_data[end_idx]-y_data[end_idx-1];
		y/=(x_data[end_idx]-x_data[end_idx-1]);
		y*=(x-x_data[end_idx]);
		y+=y_data[end_idx];
	}else if(x<=x_data[0]){
		//y=y_data[0]+(y_data[1]-y_data[0])/(x_data[1]-x_data[0])*(x-x_data[0]);
		y=y_data[1]-y_data[0];
		y/=(x_data[1]-x_data[0]);
		y*=(x-x_data[0]);
		y+=y_data[0];
	}else{
		for(i=0;i<size;i++){
			if((x_data[i]<x)&(x<=x_data[i+1])){
				//y=y_data[i]+(y_data[i+1]-y_data[i])/(x_data[i+1]-x_data[i])*(x-x_data[i]);
				y=y_data[i+1]-y_data[i];
				y/=(x_data[i+1]-x_data[i]);
				y*=(x-x_data[i]);
				y+=y_data[i];
				break;
			}
		}
	}

	return y;
}
////////////////////////////////////////////////////////////////////////
__host__ __device__ Real htoT(Real enthalpy,uint_t p_type)
{
	// corium data
	Real x_data1[5]={-554932,215789,757894,905263,1268421};
	Real y_data1[5]={280,1537,2300,2450,2650};

	Real y;
	int index0,table_size0;

	p_type=abs(p_type);

	if(k_prop_table){
		index0=k_table_index[p_type];
		table_size0=k_table_size[p_type];

		y=interp2(&k_Tab_h[index0],&k_Tab_T[index0],table_size0,enthalpy);
	}
	else
	{
		y=interp2(x_data1,y_data1,5,enthalpy);
	}

	return y;
}
////////////////////////////////////////////////////////////////////////
__host__ __device__ Real Ttoh(Real temp,uint_t p_type)
{
	// corium data
	Real y_data1[5]={-554932,215789,757894,905263,1268421};
	Real x_data1[5]={280,1537,2300,2450,2650};

	Real y;
	int index0,table_size0;

	p_type=abs(p_type);

	if(k_prop_table){
		index0=k_table_index[p_type];
		table_size0=k_table_size[p_type];

		y=interp2(&k_Tab_T[index0],&k_Tab_h[index0],table_size0,temp);
	}
	else
	{
		y=interp2(x_data1,y_data1,5,temp);
	}

	return y;
}

////////////////////////////////////////////////////////////////////////
__host__ __device__ Real viscosity(Real temp, Real D,uint_t p_type)
{
	Real vis;
	int index0,table_size0;

	p_type=abs(p_type);

	if(k_prop_table){
		index0=k_table_index[p_type];
		table_size0=k_table_size[p_type];

		vis=interp2(&k_Tab_T[index0],&k_Tab_vis[index0],table_size0,temp);
	}
	else{
			vis=1.09e-3;
	}

	if(k_viscosity_type==1){
	Real K = 1.0;
	Real N = 4.0;
	Real ty = 0;
	Real alpha = 100.0;
	
	if(abs(p_type)==2){
		Real t = vis*D;
		if(t<=ty)	vis=0;
		if(t>ty)	vis=ty/(D+1e-10)+vis*pow(D+1e-10,N-1);
	}
}

	return vis;
}
////////////////////////////////////////////////////////////////////////
__host__ __device__ Real specific_heat(Real temp, uint_t p_type)
{
	Real cpi;
	int index0,table_size0;

	p_type=abs(p_type);

	if(k_prop_table){
		index0=k_table_index[p_type];
		table_size0=k_table_size[p_type];

		cpi=interp2(&k_Tab_T[index0],&k_Tab_cp[index0],table_size0,temp);
	}else{
			cpi=4200;
	}

	return cpi;
}
////////////////////////////////////////////////////////////////////////
__host__ __device__ Real conductivity(Real temp,uint_t p_type)
{
	Real cond;
	int index0,table_size0;

	p_type=abs(p_type);

	if(k_prop_table){
		index0=k_table_index[p_type];
		table_size0=k_table_size[p_type];

		cond=interp2(&k_Tab_T[index0],&k_Tab_k[index0],table_size0,temp);
	}else{
			cond=1.65*200;
	}

	return cond;
}
////////////////////////////////////////////////////////////////////////
__host__ __device__ Real sigma(Real temp,uint_t p_type)
{
	Real y;

	p_type=abs(p_type);

  y=0.5;

	return y;
}
////////////////////////////////////////////////////////////////////////
__host__ __device__ Real diffusion_coefficient(Real temp,uint_t p_type)
{
	Real y;

	p_type=abs(p_type);

	y=0;

	return y;
}
////////////////////////////////////////////////////////////////////////
__host__ __device__ Real thermal_expansion(Real temp,uint_t p_type)
{
	Real y;

	p_type=abs(p_type);

	y=1.0/200.0;

	return y;
}
////////////////////////////////////////////////////////////////////////
__host__ __device__  Real DEVICE_clc_heat_source(Real temp,Real lbl_surf,uint_t p_type)
{
	Real y=0.0;
	Real C0=5.67e-8;
	Real eps=0.8;

	if(lbl_surf>0.5){
		y=-C0*eps*temp*temp*temp*temp*10;
		//y=0.;
	}
	return y;
}
////////////////////////////////////////////////////////////////////////
__host__ __device__  Real DEVICE_clc_heat_generation(Real temp,uint_t p_type)
{
	Real qs=0.0;

	p_type=abs(p_type);

	if(p_type==IVR_CORIUM){
		qs=6700000.0;			// volumetric heat generation rate [W/m^3]
	}

	if(p_type==MCCI_CORIUM){
		qs=1.29e9;			// volumetric heat generation rate [W/m^3]
	}

	return qs;
}
////////////////////////////////////////////////////////////////////////
__host__ __device__  Real DEVICE_clc_boiling_h(Real temp,Real lbl_surf,uint_t p_type)
{
	Real y=0.;
	Real T_sat=373.15;

	p_type=abs(p_type);

	if ((lbl_surf > 0.5) & (p_type == IVR_VESSEL))
	{
		y=-2.54e05*(8.20e-02*(temp-T_sat))*(8.20e-02*(temp-T_sat))*(8.20e-02*(temp-T_sat));
	}

	return y;
}
////////////////////////////////////////////////////////////////////////
/*
__global__ void KERNEL_find_psedo_max(part1*P1,part2*P2,Real*prove)
{
	__shared__ Real cache1[1024];

	cache1[threadIdx.x]=0;

	uint_t i=blockIdx.x;
	int_t cache_idx=threadIdx.x;

	int id=k_num_part/1024*i;
	cache1[id]=P1[i].ux*P1[i].ux+P1[i].uy*P1[i].uy+P1[i].uz*P1[i].uz;

	__syncthreads();
	uint_t s;
	for(s=blockDim.x*0.5;s>0;s>>=1){
		if(cache_idx<s){
			cache1[cache_idx]=fmax(cachex[cache_idx],cachex[cache_idx+s]);
		}
		__syncthreads();
	}
	if(cache_idx==0){
		&prove=cachex[0];
	}
}
//*/
////////////////////////////////////////////////////////////////////////
__host__ __device__  Real K_to_eta(Real tK_stiff)
{
	// corium data
	Real x_data1[5]={0,1000,5000,25000,100000};
	Real y_data1[5]={1.0,1.8,2.3,2.5,2.5};

	Real y=1;

	y=interp2(x_data1,y_data1,5,tK_stiff);

	return y;
}
