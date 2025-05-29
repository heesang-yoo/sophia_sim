////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_projection(int_t tcount, Real tdt,Real ttime,part1*P1,part2*P2,part3*P3)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part3) return;
	if(P1[i].i_type>i_type_crt) return;
	if(P1[i].p_type>=1000)	return;		// Immersed Boundary Method
	if((k_solver_type==Icsph)&&(P1[i].p_type==2))	return;		// Immersed Boundary Method

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

			tuxp=tux0+tdux_dt0*(t_dt);						// Predict x-directional velocity (dux_dt0 : acceleration of before time step)
			tuyp=tuy0+tduy_dt0*(t_dt);						// Predict y-directional velocity (duy_dt0 : acceleration of before time step)
			tuzp=tuz0+tduz_dt0*(t_dt);						// Predict z-directional velocity (duz_dt0 : acceleration of before time step)
		
			txp=tx0+(P1[i].dux+tuxp)*(t_dt)*(p_typei>0);									// Predict x-directional position (ux0 : velocity of before time step)
			typ=ty0+(P1[i].duy+tuyp)*(t_dt)*(p_typei>0);
			tzp=tz0+tuzp*(t_dt)*(p_typei>0)*(P1[i].eliz);									// Predict z-directional position (ux0 : velocity of before time step)

		}else{
			txp=tx0;typ=ty0;tzp=tz0;
			tuxp=tux0;tuyp=tuy0;tuzp=tuz0;
			tuxp=P1[i].ux;
			tuyp=P1[i].uy;
			tuzp=P1[i].uz;
		}

		P1[i].x=tx0;															// Update particle data by predicted x-directional position
		P1[i].y=ty0;															// Update particle data by predicted y-directional position
		P1[i].z=tz0;															// Update particle data by predicted z-directional position
		P1[i].x_star=txp;															// Update particle data by predicted x-directional position
		P1[i].y_star=typ;															// Update particle data by predicted y-directional position
		P1[i].z_star=tzp;															// Update particle data by predicted z-directional position
		P1[i].ux=tux0;														// Update particle data by predicted x-directional velocity
		P1[i].uy=tuy0;														// Update particle data by predicted y-directional velocity
		P1[i].uz=tuz0;														// Update particle data by predicted z-directional velocity

		P2[i].x0=tx0;															// update x-directional position
		P2[i].y0=ty0;															// update y-directional position
		P2[i].z0=tz0;															// update z-directional position
		P2[i].ux0=tux0;														// update x-directional velocity
		P2[i].uy0=tuy0;														// update y-directional velocity
		P2[i].uz0=tuz0;														// update z-directional velocity
	}

	if((k_rho_type==Continuity)&&(P1[i].p_type==2)&&(k_solver_type==Icsph)){
		Real trho=P1[i].rho;
		if(tcount==0)	P1[i].vol0 = P1[i].m/trho;
		P2[i].rho0=trho;
		P1[i].rho=trho;
	}else{
		Real trho=P1[i].rho;
		if(tcount==0)	P1[i].vol0 = P1[i].m/trho;
		P2[i].rho0=trho;
	}
	if(ttime==0){
		P2[i].rho_ref = 1000.0;
		if(P1[i].p_type==2)	P2[i].rho_ref = 1.0;
	}	
	if((k_con_solve==1)&&(P1[i].p_type>0)){
		if(enthalpy_eqn){
			Real tenthalpyp=P1[i].enthalpy;
			P2[i].enthalpy0=tenthalpyp;

			tenthalpyp+=P3[i].denthalpy*(t_dt)*(P1[i].p_type!=-1);
			P1[i].enthalpy=tenthalpyp;
			P1[i].temp=htoT(tenthalpyp,p_typei);
			}else{
			Real ttemp=P1[i].temp;
			P2[i].temp0=ttemp;
			P1[i].temp=ttemp+P3[i].dtemp*(t_dt)*(P1[i].p_type!=-1);
			}
	}

	if(k_concn_solve==1){
		Real tconcn=P1[i].concn;
		P2[i].concn0=tconcn;
		P1[i].concn=tconcn+P3[i].dconcn*(t_dt);
	}

}
////////////////////////////////////////////////////////////////////////
// corrector step for Predictor-Corrector time integration
__global__ void KERNEL_time_update_projection(const Real tdt,part1*P1,part1*TP1,part2*P2,part2*TP2,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part3) return;
	if(P1[i].i_type>i_type_crt) return;
	if(P1[i].p_type>=1000)	return;		// Immersed Boundary Method
	if((k_solver_type==Icsph)&&(P1[i].p_type==2))	return;		// Immersed Boundary Method

	Real tx0,ty0,tz0,xc,yc,zc;									// position
	Real tux0,tuy0,tuz0,uxc,uyc,uzc;						// velocity
	Real tux,tuy,tuz;														// velocity
	Real dux_dt,duy_dt,duz_dt;									// accleration (time derivative of velocity)
	Real t_dt=tdt;

	int_t buffer_type=P1[i].buffer_type;
	int_t p_type_i=P1[i].p_type;

	if(p_type_i==MOVING){
		tx0=P1[i].x;														// x-directional initial position
		ty0=P1[i].y;														// x-directional initial position
		tz0=P1[i].z;														// x-directional initial position
		tux=P1[i].ux;
		tuy=P1[i].uy;
		tuz=P1[i].uz;

		TP1[i].x=xc;															// update x-directional position
		TP1[i].y=yc;															// update y-directional position
		TP1[i].z=zc;															// update z-directional position
		TP1[i].ux=tux;														// update x-directional velocity
		TP1[i].uy=tuy;														// update y-directional velocity
		TP1[i].uz=tuz;														// update z-directional velocity

	}else{
		tx0=P1[i].x;														// x-directional initial position
		ty0=P1[i].y;														// x-directional initial position
		tz0=P1[i].z;														// x-directional initial position
		if(p_type_i>0){
			tux0=P1[i].ux;						// x-directional initial velocity
			tuy0=P1[i].uy;						// y-directional initial velocity
			tuz0=P1[i].uz;						// z-directional initial velocity

			dux_dt=(P3[i].ftotalx+P3[i].fpx)*(buffer_type==0)+P1[i].fbx/P1[i].rho;			// x-directional acceleration
			duy_dt=(P3[i].ftotaly+P3[i].fpy)*(buffer_type==0)+P1[i].fby/P1[i].rho;			// y-directional acceleration
			duz_dt=(P3[i].ftotalz+P3[i].fpz)*(buffer_type==0)+P1[i].fbz/P1[i].rho;			// z-directional acceleration
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

		// Euler method
		// xc=tx0+uxc*(t_dt)*(p_type_i>0)*(P1[i].elix);												// correct x-directional position
		// yc=ty0+uyc*(t_dt)*(p_type_i>0)*(P1[i].eliy);												// correct Y-directional position
		// zc=tz0+uzc*(t_dt)*(p_type_i>0)*(P1[i].eliz);
		
		xc=tx0+(P1[i].dux+(P2[i].ux0+uxc)/2.0)*(t_dt)*(p_type_i>0)*(P1[i].elix);												// correct x-directional position
		yc=ty0+(P1[i].duy+(P2[i].uy0+uyc)/2.0)*(t_dt)*(p_type_i>0)*(P1[i].eliy);												// correct Y-directional position
		zc=tz0+(P2[i].uz0+uzc)/2.0*(t_dt)*(p_type_i>0)*(P1[i].eliz);

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
	if((k_rho_type==Continuity)&&(P1[i].p_type==2)&&(k_solver_type==Icsph)) TP1[i].rho=P2[i].rho0+P3[i].drho*t_dt;
	else TP1[i].rho=P1[i].rho;

	// 경계 변수 정의 (초기화는 시뮬레이션 시작 시)
	Real x_min0 = -0.3;
	Real x_max0 = 0.3;
	Real y_min0 = 0.0;
	Real y_max0 = 0.6;

	// 마진 거리
	Real margin = P1[i].h / 1.5 / 2.;

	// // 입자 경계 확인 및 반사 처리
	if ((TP1[i].x < x_min0 - margin || TP1[i].x > x_max0 + margin ||
		TP1[i].y < y_min0 - margin || TP1[i].y > y_max0 + margin) &&
		TP1[i].p_type > 0)
	{
		// TP1[i].i_type = 3;

	// 	// X 방향 반사
	// 	if (TP1[i].x < x_min0) {
	// 		TP1[i].x = x_min0 + (x_min0 - TP1[i].x);  // 반사 위치
	// 		TP1[i].ux *= -1.0;
	// 	} else if (TP1[i].x > x_max0) {
	// 		TP1[i].x = x_max0 - (TP1[i].x - x_max0);
	// 		TP1[i].ux *= -1.0;
	// 	}

	// 	// Y 방향 반사
	// 	if (TP1[i].y < y_min0) {
	// 		TP1[i].y = y_min0 + (y_min0 - TP1[i].y);
	// 		TP1[i].uy *= -1.0;
	// 	} else if (TP1[i].y > y_max0) {
	// 		TP1[i].y = y_max0 - (TP1[i].y - y_max0);
	// 		TP1[i].uy *= -1.0;
	// 	}
	}


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
	TP1[i].dpres=P1[i].dpres;
	TP1[i].flt_s=P1[i].flt_s;
	TP1[i].m=P1[i].m;
	TP1[i].ncell=P1[i].ncell;
	TP1[i].h=P1[i].h;
	TP1[i].grad_rhox=P1[i].grad_rhox;
	TP1[i].grad_rhoy=P1[i].grad_rhoy;
	TP1[i].grad_rhoz=P1[i].grad_rhoz;
	TP1[i].k_turb=P1[i].k_turb;
	TP1[i].e_turb=P1[i].e_turb;

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

	TP1[i].PPE1=P1[i].PPE1;
	TP1[i].PPE2=P1[i].PPE2;
}
////////////////////////////////////////////////////////////////////////

// Projection step: computes pressure correction or velocity projection
void projection(
    dim3 b, dim3 t,
    part1* dev_SP1, part2* dev_SP2, part3* dev_P3
) {
    KERNEL_clc_projection<<<b, t>>>(count, dt, time, dev_SP1, dev_SP2, dev_P3);
    cudaDeviceSynchronize();
}

// Time update after projection step: update particle variables for the next time step
void timeUpdateProjection(
    dim3 b, dim3 t,
    part1* dev_SP1, part1* dev_P1,
    part2* dev_SP2, part2* dev_P2,
    part3* dev_P3
) {
    KERNEL_time_update_projection<<<b, t>>>(dt, dev_SP1, dev_P1, dev_SP2, dev_P2, dev_P3);
    cudaDeviceSynchronize();
}
