////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_preparation(Real tdt,Real ttime,part1*P1,part2*P2,part3*P3)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part3) return;
	if(P1[i].i_type>i_type_crt) return;
	if((k_solver_type==Icsph)&&(P1[i].p_type==1))	return;		// Immersed Boundary Method

	int_t p_typei;
	Real tx0,ty0,tz0,txp,typ,tzp;
	Real tux0,tuy0,tuz0,tuxp,tuyp,tuzp;
	Real tdux_dt0,tduy_dt0,tduz_dt0;
	Real t_dt;

	int_t buffer_type=P1[i].buffer_type;

	t_dt=tdt;
	p_typei=P1[i].p_type;

	if(p_typei==MOVING){
		P2[i].x0=P1[i].x;
		P2[i].y0=P1[i].y;
		P2[i].z0=P1[i].z;

		P1[i].ux=k_ball_vel;
		P1[i].uy=0;
		P1[i].uz=0;
		P2[i].ux0=k_ball_vel;
		P2[i].uy0=0;
		P2[i].uz0=0;
	}else{

		P2[i].x0=P1[i].x;															// update x-directional position
		P2[i].y0=P1[i].y;															// update y-directional position
		P2[i].z0=P1[i].z;															// update z-directional position
		P2[i].ux0=P1[i].ux;														// update x-directional velocity
		P2[i].uy0=P1[i].uy;														// update y-directional velocity
		P2[i].uz0=P1[i].uz;														// update z-directional velocity
	}

	if(ttime==0){
		P2[i].rho_ref = 1000.0;
		if(P1[i].p_type==2)	P2[i].rho_ref = 1.0;
		P1[i].vol0 = P1[i].m/P1[i].rho;
		P1[i].vol = P1[i].vol0;

	}	

	if((k_rho_type==Continuity)&&(P1[i].p_type>0)){
		Real trho=P1[i].rho;
		P2[i].rho0=trho;
		P1[i].rho=trho;
	}

	// // KERNEL_clc_reference_density
	//----------------------------------------------------
	// KERNEL_clc_predictor_enthalpy - Update particle data by predicted density
	if(k_con_solve==1&&(P1[i].p_type>0)){
		P2[i].enthalpy0=P1[i].enthalpy;
		P1[i].temp=htoT(P1[i].enthalpy,p_typei);
	}

	// KERNEL_clc_predictor_concn - Predict  (dconcn_dt0 : time derivatve of density of before time step)
	// Update particle data by predicted
	if(k_concn_solve==1&&(P1[i].p_type>0)){
		P2[i].concn0=P1[i].concn;
	}

}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_predictor(int_t tcount, Real tdt,Real ttime,part1*P1,part2*P2,part3*P3)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part3) return;
	if(P1[i].i_type>i_type_crt) return;
	if(P1[i].p_type>=1000)	return;		// Immersed Boundary Method
	if((k_solver_type==Icsph)&&(P1[i].p_type==1))	return;		// Immersed Boundary Method

	int_t p_typei;
	Real tx0,ty0,tz0,txp,typ,tzp;
	Real tux0,tuy0,tuz0,tuxp,tuyp,tuzp;
	Real tdux_dt0,tduy_dt0,tduz_dt0;
	Real t_dt;

	int_t buffer_type=P1[i].buffer_type;

	t_dt=tdt;
	p_typei=P1[i].p_type;

	if(p_typei==MOVING){
		P2[i].x0=P1[i].x;
		P2[i].y0=P1[i].y;
		P2[i].z0=P1[i].z;

		// P1[i].ux=-0.06*PI*sin(ttime*PI);
		P1[i].ux=k_ball_vel;
		P1[i].uy=0;
		P1[i].uz=0;
		P2[i].ux0=k_ball_vel;
		P2[i].uy0=0;
		P2[i].uz0=0;
	}else{
		tx0=P1[i].x;															// initial x-directional position
		ty0=P1[i].y;															// initial y-directional position
		tz0=P1[i].z;															// initial z-directional position
		tux0=P1[i].ux;															// initial x-directional position //YHS
		tuy0=P1[i].uy;															// initial y-directional position //YHS
		tuz0=P1[i].uz;															// initial z-directional position //YHS
		if(p_typei>=0){
			tux0=P1[i].ux;													// initial x-directional velocity
			tuy0=P1[i].uy;													// initial y-directional velocity
			tuz0=P1[i].uz;													// initial z-directional velocity

			tdux_dt0=P3[i].ftotalx*(buffer_type==0);									// initial x-directional acceleration
			tduy_dt0=P3[i].ftotaly*(buffer_type==0);									// initial y-directional acceleration
			tduz_dt0=P3[i].ftotalz*(buffer_type==0);									// initial z-directional acceleration

			txp=tx0+(P1[i].dux+tux0*P1[i].elix)*(t_dt*0.5)*(p_typei>0);									// Predict x-directional position (ux0 : velocity of before time step)
			typ=ty0+(P1[i].duy+tuy0*P1[i].eliy)*(t_dt*0.5)*(p_typei>0);
			// txp=tx0+(tux0)*(t_dt*0.5)*(p_typei>0);									// Predict x-directional position (ux0 : velocity of before time step)
			// typ=ty0+(tuy0)*(t_dt*0.5)*(p_typei>0);

			tzp=tz0+tuz0*(t_dt*0.5)*(p_typei>0)*(P1[i].eliz);									// Predict z-directional position (ux0 : velocity of before time step)

			tuxp=tux0+tdux_dt0*(t_dt*0.5);						// Predict x-directional velocity (dux_dt0 : acceleration of before time step)
			tuyp=tuy0+tduy_dt0*(t_dt*0.5);						// Predict y-directional velocity (duy_dt0 : acceleration of before time step)
			tuzp=tuz0+tduz_dt0*(t_dt*0.5);						// Predict z-directional velocity (duz_dt0 : acceleration of before time step)
		}else{
			txp=tx0;typ=ty0;tzp=tz0;
						tuxp=tux0;tuyp=tuy0;tuzp=tuz0;
			tuxp=P1[i].ux;
			tuyp=P1[i].uy;
			tuzp=P1[i].uz;
		}

		P1[i].x=txp;															// Update particle data by predicted x-directional position
		P1[i].y=typ;															// Update particle data by predicted y-directional position
		P1[i].z=tzp;															// Update particle data by predicted z-directional position
		P1[i].ux=tuxp;														// Update particle data by predicted x-directional velocity
		P1[i].uy=tuyp;														// Update particle data by predicted y-directional velocity
		P1[i].uz=tuzp;														// Update particle data by predicted z-directional velocity

		P2[i].x0=tx0;															// update x-directional position
		P2[i].y0=ty0;															// update y-directional position
		P2[i].z0=tz0;															// update z-directional position
		P2[i].ux0=tux0;														// update x-directional velocity
		P2[i].uy0=tuy0;														// update y-directional velocity
		P2[i].uz0=tuz0;														// update z-directional velocity
	}

	if(ttime==0){
		P2[i].rho_ref = 1000.0;
		if(P1[i].p_type==2)	P2[i].rho_ref = 1.0;

	}	
	
	if((k_rho_type==Continuity)&&(P1[i].p_type>0)&&(P3[i].lbl_surf!=3)){
		Real trho=P1[i].rho;
		if(tcount==0)	P1[i].vol0 = P1[i].m/trho;
		P2[i].rho0=trho;
		P1[i].rho=trho+P3[i].drho*(t_dt*0.5);
	}else{
		Real trho=P1[i].rho;
		if(tcount==0){
			P1[i].vol0 = P1[i].m/trho;
			P1[i].vol = P1[i].vol0;
		}
		P2[i].rho0=trho;
	}

	if((k_con_solve==1)&&(P1[i].p_type>0)){
		if(enthalpy_eqn){
			Real tenthalpyp=P1[i].enthalpy;
			P2[i].enthalpy0=tenthalpyp;

			tenthalpyp+=P3[i].denthalpy*(t_dt*0.5)*(P1[i].p_type!=-1);
			P1[i].enthalpy=tenthalpyp;
			P1[i].temp=htoT(tenthalpyp,p_typei);
			}else{
			Real ttemp=P1[i].temp;
			P2[i].temp0=ttemp;
			P1[i].temp=ttemp+P3[i].dtemp*(t_dt*0.5)*(P1[i].p_type!=-1);
			}
	}

	if(k_concn_solve==1){
		Real tconcn=P1[i].concn;
		P2[i].concn0=tconcn;
		P1[i].concn=tconcn+P3[i].dconcn*(t_dt*0.5);
	}

}
////////////////////////////////////////////////////////////////////////
// corrector step for Predictor-Corrector time integration
__global__ void KERNEL_time_update_single(const Real tdt,part1*P1,part1*TP1,part2*P2,part2*TP2,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	// if(i>=k_num_part3) return;
	// if(P1[i].i_type!=inout) return;

	if(i>=k_num_part3) return;
	if(P1[i].i_type>i_type_crt) return;
	if(P1[i].p_type>=1000)	return;		// Immersed Boundary Method
	if((k_solver_type==Icsph)&&(P1[i].p_type==1))	return;		// Immersed Boundary Method

	Real tx0,ty0,tz0,xc,yc,zc;									// position
	Real tux0,tuy0,tuz0,uxc,uyc,uzc;						// velocity
	Real tux,tuy,tuz;														// velocity
	Real dux_dt,duy_dt,duz_dt;									// accleration (time derivative of velocity)
	Real t_dt=tdt;

	int_t buffer_type=P1[i].buffer_type;
	int_t p_type_i=P1[i].p_type;

	if(p_type_i==MOVING){
		tx0=P2[i].x0;														// x-directional initial position
		ty0=P2[i].y0;														// x-directional initial position
		tz0=P2[i].z0;														// x-directional initial position
		tux=P2[i].ux0;
		tuy=P2[i].uy0;
		tuz=P2[i].uz0;

		// xc=tx0+tux*(t_dt);												// correct x-directional position
		// yc=ty0+tuy*(t_dt);												// correct Y-directional position
		// zc=tz0+tuz*(t_dt);												// correct Z-directional position

		TP1[i].x=xc;															// update x-directional position
		TP1[i].y=yc;															// update y-directional position
		TP1[i].z=zc;															// update z-directional position
		TP1[i].ux=tux;														// update x-directional velocity
		TP1[i].uy=tuy;														// update y-directional velocity
		TP1[i].uz=tuz;														// update z-directional velocity

	}else{
		tx0=P2[i].x0;														// x-directional initial position
		ty0=P2[i].y0;														// x-directional initial position
		tz0=P2[i].z0;														// x-directional initial position
		if(p_type_i>0){
			tux0=P2[i].ux0;						// x-directional initial velocity
			tuy0=P2[i].uy0;						// y-directional initial velocity
			tuz0=P2[i].uz0;						// z-directional initial velocity

			// dux_dt=P3[i].ftotalx*(buffer_type==0);			// x-directional acceleration
			// duy_dt=P3[i].ftotaly*(buffer_type==0);			// y-directional acceleration
			// duz_dt=P3[i].ftotalz*(buffer_type==0);			// z-directional acceleration
			dux_dt=P3[i].ftotalx*(buffer_type==0)+P1[i].fbx/P1[i].rho;			// x-directional acceleration
			duy_dt=P3[i].ftotaly*(buffer_type==0)+P1[i].fby/P1[i].rho;			// y-directional acceleration
			duz_dt=P3[i].ftotalz*(buffer_type==0)+P1[i].fbz/P1[i].rho;			// z-directional acceleration
		}else{
			tux0=P2[i].ux0;
			tuy0=P2[i].uy0;
			tuz0=P2[i].uz0;
			dux_dt=duy_dt=duz_dt=0.0;
		}

		uxc=tux0+dux_dt*(t_dt);										// correct x-directional velocity
		uyc=tuy0+duy_dt*(t_dt);										// correct y-directional velocity
		uzc=tuz0+duz_dt*(t_dt);										// correct z-directional velocity

		if((uxc*uxc+uyc*uyc+uzc*uzc)>=k_u_limit*k_u_limit){
			uxc=tux0;
			uyc=tuy0;
			uzc=tuz0;
		}
		xc=tx0+(P1[i].dux+uxc*P1[i].elix)*(t_dt)*(p_type_i>0);												// correct x-directional position
		yc=ty0+(P1[i].duy+uyc*P1[i].eliy)*(t_dt)*(p_type_i>0);												// correct Y-directional position
		zc=tz0+uzc*(t_dt)*(p_type_i>0)*(P1[i].eliz);												// correct Z-directional position

		if(!k_xsph_solve){
				TP1[i].x=xc;															// update x-directional position
				TP1[i].y=yc;															// update y-directional position
				TP1[i].z=zc;															// update z-directional position
		}

		TP1[i].ux=uxc;														// update x-directional velocity
		TP1[i].uy=uyc;														// update y-directional velocity
		TP1[i].uz=uzc;														// update z-directional velocity
	}

	//KERNEL_clc_precor_update_continuity ---------------------
	if((k_rho_type==Continuity)&&(P1[i].p_type>0)&&(P3[i].lbl_surf!=3)) TP1[i].rho=P2[i].rho0+P3[i].drho*t_dt;
	else TP1[i].rho=P1[i].rho;
	if(((P1[i].rho-P2[i].rho_ref)>P2[i].rho_ref*0.05)&(P3[i].drho>0))	TP1[i].rho=P2[i].rho0;
	if(((P1[i].rho-P2[i].rho_ref)<-P2[i].rho_ref*0.05)&(P3[i].drho<0))	TP1[i].rho=P2[i].rho0;
	if((TP1[i].x<-1.2-P1[i].h/1.5||TP1[i].x>2.0196+P1[i].h/1.5||TP1[i].y<0-P1[i].h/1.5||TP1[i].y>1.8+P1[i].h/1.5)&&TP1[i].p_type>0)	TP1[i].i_type=3;
	
	//update_properties_enthalpy-------------------------------
	if((k_con_solve==1)&&(P1[i].p_type>0)){
		if(enthalpy_eqn){
			TP1[i].enthalpy=P2[i].enthalpy0+P3[i].denthalpy*t_dt*(P1[i].p_type!=-1);
			TP1[i].temp=P1[i].temp;
		}else{
			TP1[i].temp=P2[i].temp0+P3[i].dtemp*t_dt*(P1[i].p_type!=-1);
		}
	}else{
		TP1[i].enthalpy=P1[i].enthalpy;
		TP1[i].temp=P1[i].temp;
	}

	//update_properties_concn----------------------------------
	if(k_concn_solve==1) TP1[i].concn=P2[i].concn0+P3[i].dconcn*t_dt;
	else TP1[i].concn=P1[i].concn;

	TP1[i].pres=P1[i].pres;
	TP1[i].flt_s=P1[i].flt_s;
	TP1[i].m=P1[i].m;
	TP1[i].ncell=P1[i].ncell;
	TP1[i].h=P1[i].h;
	TP1[i].grad_rhox=P1[i].grad_rhox;
	TP1[i].grad_rhoy=P1[i].grad_rhoy;
	TP1[i].grad_rhoz=P1[i].grad_rhoz;
	TP1[i].k_turb=P1[i].k_turb;
	TP1[i].e_turb=P1[i].e_turb;
	TP1[i].D=P1[i].D;
	TP1[i].elix=P1[i].elix;
	TP1[i].eliy=P1[i].eliy;
	TP1[i].shiftx=P1[i].shiftx;
	TP1[i].shifty=P1[i].shifty;
	TP1[i].vortx=P1[i].vortx;
	TP1[i].vorty=P1[i].vorty;
	TP1[i].vortz=P1[i].vortz;
	TP1[i].vol=P1[i].vol;
	TP1[i].vol0=P1[i].vol0;
	TP2[i].rho_ref=P2[i].rho_ref;
	TP1[i].flt_s=P1[i].flt_s;
	TP1[i].fcx = P1[i].fcx;
	TP1[i].fcy = P1[i].fcy;
	TP1[i].pos=P1[i].pos;
	TP1[i].concentration=P1[i].concentration;
}
////////////////////////////////////////////////////////////////////////

#define round(x) ((x)>=0?(long)((x)+0.5):(long)((x)-0.5))
// __host__ __device__ int clc_idx_insert(Real xc, Real yc, Real zc, Real tdelx, int Nsx_, int Nsz_)
// {
// 	int Ix,Iy, Iz;
// 	int idx_1D;

// 	Ix = round((xc-k_x_min)/tdelx);
// 	Iy = round((yc-k_y_min)/tdelx);
// 	Iz = round((zc-k_z_min)/tdelx);

// 	if (k_dim==2) idx_1D=(k_num_part3-1)-(Iy);
// 	if (k_dim==3) idx_1D=(k_num_part3-1)-(Iz*Nsx_+Ix);

// 	return idx_1D;
// }
__host__ __device__ int clc_idx_insert(Real xc, Real zc, Real tdelx, int Nsx_, int Nsz_)
{
	int Ix, Iz;
	int idx_1D;

	Ix = round((xc-k_x_min)/tdelx);
	Iz = round((zc-k_z_min)/tdelx);

	if (k_dim==2) idx_1D=(k_num_part2-1)-(Ix);
	if (k_dim==3) idx_1D=(k_num_part2-1)-(Iz*Nsx_+Ix);

	return idx_1D;
}
////////////////////////////////////////////////////////////////////////
// open_boundary
__global__ void KERNEL_time_update_buffer(const Real tdt,part1*P1,part1*TP1,part2*P2,part3*P3,Real space_,int Nsx_, int Nsz_)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>2) return;
	if(P1[i].p_type<1) return;

	Real tx0,ty0,tz0,xc,yc,zc;									// position
	Real tux0,tuy0,tuz0,uxc,uyc,uzc;						// velocity
	Real t_dt=tdt;
	Real tmp_h=P1[i].h;
	int_t idx_insert;

	int_t p_type=P1[i].p_type;
	int_t i_type=P1[i].i_type;
	int_t buffer_type=P1[i].buffer_type;

	xc=TP1[i].x;												// correct x-directional position
	yc=TP1[i].y;												// correct Y-directional position
	zc=TP1[i].z;												// correct Z-directional position

	// inlet
	if ((buffer_type==1)&&(i_type==2)) {
		if (yc>=L2) {
			TP1[i].i_type=1;
			TP1[i].buffer_type=0;

			idx_insert=clc_idx_insert(xc,zc,tmp_h/h_coeff,Nsx_,Nsz_);

			TP1[idx_insert]=P1[i];
			TP1[idx_insert].x=xc;
			TP1[idx_insert].y=L1;
			TP1[idx_insert].z=zc;

			if (k_rho_type==Mass_Sum) TP1[i].flt_s=1.0;
		}
	}

	if (yc>=L3){
		TP1[i].buffer_type=2;
		TP1[i].i_type=2;
	}
	if (yc>=L4){
		TP1[i].i_type=3;
	}

	if (TP1[i].buffer_type==0 && yc<L1){
		TP1[i].i_type=3;
	}
}
