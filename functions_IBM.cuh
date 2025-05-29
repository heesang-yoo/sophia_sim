////////////////////////////////////////////////////////////////////////
__global__ void IBM_predictor(Real t_dt,Real ttime,part1*P1,part2*P2)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(P1[i].p_type<1000) return;

	int_t p_typei;
	Real tx0,ty0,tz0,txp,typ,tzp;
	Real tux0,tuy0,tuz0,tuxp,tuyp,tuzp;
	Real tdux_dt0,tduy_dt0,tduz_dt0;

	// P1[i].ux=0.0;
	// // P1[i].uy=0.0;
	// P1[i].uy=0.02*sin(2.0*PI/1.5*ttime);
	// // P2[i].uz0=P1[i].uz;

	tx0 = P1[i].x;
	ty0 = P1[i].y;
	tz0 = P1[i].z;

	tux0= P1[i].ux;
	tuy0= P1[i].uy;
	tuz0= P1[i].uz;

	tdux_dt0=P1[i].ax;
	tduy_dt0=P1[i].ay;
	tduz_dt0=P1[i].az;

	//update position

	txp = tx0 + (tux0*0.5)*t_dt;
	typ = ty0 + (tuy0*0.5)*t_dt;
	tzp = tz0 + (tuz0*0.5)*t_dt;

	// Static point

	// if(P1[i].ccc==3){
	// 	tux0=0.0;
	// 	tuy0=0.0;
	// 	tuz0=0.0;
	// 	tdux_dt0=tduy_dt0=tduz_dt0=0.0;
	// }

	//update velocity

	tuxp = tux0 + (tdux_dt0*0.5)*t_dt;
	tuyp = tuy0 + (tduy_dt0*0.5)*t_dt;
	tuzp = tuz0 + (tduz_dt0*0.5)*t_dt;


	P1[i].x = txp;
	P1[i].y = typ;
	P1[i].z = tzp;

	P1[i].ux = tuxp;
	P1[i].uy = tuyp;
	P1[i].uz = tuzp;

	P2[i].x0 = tx0;
	P2[i].y0 = ty0;
	P2[i].z0 = tz0;

	P2[i].ux0 = tux0;
	P2[i].uy0 = tuy0;
	P2[i].uz0 = tuz0;

	P2[i].rho_ref=1000.0;
	if(P1[i].p_type==2000)	P2[i].rho_ref=10000.0;
	
	if(ttime==0)	P1[i].vol0 = P1[i].m/P1[i].rho;
	if(ttime==0)	P1[i].vol = P1[i].m/P1[i].rho;

	// if((P1[i].p_type>=2000)&&(ttime>0))	P1[i].rho=P2[i].rho_ref/(P1[i].jacob+1e-20);

}
////////////////////////////////////////////////////////////////////////
__global__ void IBM_force_interpolation(Real t_dt,int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(P1[i].p_type<1000)	return;		// Immersed Boundary Method

	int_t icell,jcell;
	Real xi,yi,uxi,uyi;
	Real rhoi;
	Real grad_rhoxi,grad_rhoyi;
	Real search_range,tmp_h,tmp_A;
	// Real tmpx,tmpy,tmppx,tmppy,tmp_R;
	Real tmpux, tmpuy, filt;
	int p_type_i;
	Real ibm_length = 2.0/k_kappa;

	// Real ibm_length = 1.0;

	p_type_i=P1[i].p_type;
	xi=P1[i].x;
	yi=P1[i].y;
	uxi=P1[i].ux;
	uyi=P1[i].uy;
	rhoi=P1[i].rho;
	tmp_h=P1[i].h;
	grad_rhoxi=P1[i].grad_rhox;
	grad_rhoyi=P1[i].grad_rhoy;
	tmp_A=calc_tmpA(tmp_h*ibm_length);
	search_range=k_search_kappa*tmp_h;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;

	// tmpx=tmpy=tmppx=tmppy=tmp_R=0.0;
	tmpux=tmpuy=filt=0.0;
	// for(int_t y=-P1[i].ncell;y<=P1[i].ncell;y++){
	// 	for(int_t x=-P1[i].ncell;x<=P1[i].ncell;x++){
			for(int_t y=-4;y<=4;y++){
				for(int_t x=-4;x<=4;x++){
			int_t k=idx_cell(icell+x,jcell+y,0);

			if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))) continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					Real xj,yj,tdist;
					int itype;
					int p_type_j, buffer_typej;

					xj=P1[j].x;
					yj=P1[j].y;
					itype=P1[j].i_type;
					buffer_typej=P1[j].buffer_type;

					// if((itype!=2)|((buffer_typej!=Inlet)&(buffer_typej!=Outlet))){
					if(itype!=4){
					if(P1[j].p_type<1000){

					tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj))+1e-20;
					if(tdist<search_range*ibm_length){
						Real mj,uxj,uyj,rhoj,tmprho;
						Real hj;
						Real twij=calc_kernel_wij(tmp_A,tmp_h*ibm_length,tdist);
						hj=P1[i].h;

						// printf("%f\t",twij);
						// apply_MLS_filter_2D(P3[i].A,twij,&twij,xi,xj,yi,yj,hj);

						// printf("%f\t%f\t%f\n",P3[i].A[0][0],P3[i].A[1][0],P3[i].A[2][0]);
						// printf("%f\n",twij);
						p_type_j=P1[j].p_type;

						mj=P1[j].m;
						rhoj=P1[j].rho;
						uxj=P1[j].ux;
						uyj=P1[j].uy;

						tmprho=mj/rhoj;
						tmpux+=(uxj)*tmprho*twij;
						tmpuy+=(uyj)*tmprho*twij;
						filt+=tmprho*twij;
						}
					}
				}
			}
				}
		}
	}

	// preliminary velocity
	P1[i].concentration = filt;
	P1[i].ux_i = tmpux/filt;
	P1[i].uy_i = tmpuy/filt;

	// the force applied to solid for no-slip condition
	P1[i].fbx = 1000.0 * (P1[i].ux-tmpux/filt)/t_dt;
	P1[i].fby = 1000.0 * (P1[i].uy-tmpuy/filt)/t_dt;
	// P1[i].fbz = 1000.0 * (P1[i].uz-tmpuz)/t_dt;
}

////////////////////////////////////////////////////////////////////////
__global__ void IBM_spreading_interpolation(int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(P1[i].p_type>=1000)	return;		// Immersed Boundary Method
	if(i>=k_num_part3) return;

	int_t icell,jcell;
	Real xi,yi,uxi,uyi;
	Real rhoi;
	Real grad_rhoxi,grad_rhoyi;
	Real search_range,tmp_h,tmp_A;
	// Real tmpx,tmpy,tmppx,tmppy,tmp_R;
	Real tmpux, tmpuy,filt;
	int p_type_i;
	Real fb_x, fb_y, fb_z;
	Real ibm_length = 2.0/k_kappa;
	// Real ibm_length = 1.0;


	p_type_i=P1[i].p_type;
	xi=P1[i].x;
	yi=P1[i].y;
	uxi=P1[i].ux;
	uyi=P1[i].uy;
	rhoi=P1[i].rho;
	tmp_h=P1[i].h;
	grad_rhoxi=P1[i].grad_rhox;
	grad_rhoyi=P1[i].grad_rhoy;
	tmp_A=calc_tmpA(tmp_h*ibm_length);
	search_range=k_search_kappa*tmp_h;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;

	// tmpx=tmpy=tmppx=tmppy=tmp_R=0.0;
	tmpux=tmpuy=filt=0.0;
	fb_x = fb_y = fb_z = 0.0;
	for(int_t y=-P1[i].ncell;y<=P1[i].ncell;y++){
		for(int_t x=-P1[i].ncell;x<=P1[i].ncell;x++){
			// int_t k=(icell+x)+k_NI*(jcell+y);
			int_t k=idx_cell(icell+x,jcell+y,0);

			if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))) continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					Real xj,yj,tdist;
					int itype;
					int p_type_j, buffer_typej;

					xj=P1[j].x;
					yj=P1[j].y;
					itype=P1[j].i_type;
					buffer_typej=P1[j].buffer_type;

					// if((itype!=2)|((buffer_typej!=Inlet)&(buffer_typej!=Outlet))){
					if(itype!=4){
					if(P1[j].p_type>=1000){

					tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj))+1e-20;
					if(tdist<search_range*ibm_length){
						Real mj,tdwx,tdwy,uxj,uyj,rhoj,grad_rhoxj,grad_rhoyj,phi_ij,tmprho,tmpr;
						Real fb_xm, fb_ym, fb_zm;
						Real tdwij=calc_kernel_dwij(tmp_A,tmp_h*ibm_length,tdist);
						Real twij=calc_kernel_wij(tmp_A,tmp_h*ibm_length,tdist);


						p_type_j=P1[j].p_type;

						mj=P1[j].m;
						rhoj=P1[j].rho;
						uxj=P1[j].ux;
						uyj=P1[j].uy;
						grad_rhoxj=P1[j].grad_rhox;
						grad_rhoyj=P1[j].grad_rhoy;
						tmpr=0.0;

						fb_xm = P1[j].fbx;
						fb_ym = P1[j].fby;
						fb_zm = P1[j].fbz;

						tdwx=tdwij*(xi-xj)/tdist;
						tdwy=tdwij*(yi-yj)/tdist;

						if(k_kgc_solve>0){
							Real twij=calc_kernel_wij(tmp_A,tmp_h*ibm_length,tdist);
							apply_gradient_correction_2D(P3[i].Cm,twij,tdwx,tdwy,&tdwx,&tdwy);
						}

						// Real hj=P1[j].h;
						// apply_MLS_filter_2D(P3[i].A,twij,&twij,xi,xj,yi,yj,hj);


						tmprho=mj/rhoj;
						fb_x+=fb_xm*twij*tmprho;
						fb_y+=fb_ym*twij*tmprho;
						filt+=tmprho*twij;
						}
					}
				}
			}
				}
		}
	}

	// the force applied to fluid for no-slip condition

	P1[i].fbx = fb_x/(filt+1e-10);
	P1[i].fby = fb_y/(filt+1e-10);
	P1[i].concentration = filt;

}


////////////////////////////////////////////////////////////////////////
__global__ void IBM_corrector(Real t_dt,Real ttime,part1*P1,part1*TP1,part2*P2,part3*P3)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(P1[i].p_type<1000) return;

	int_t p_typei;
	Real tx0,ty0,tz0,txc,tyc,tzc;
	Real tux0,tuy0,tuz0,tuxc,tuyc,tuzc;
	Real tdux_dt,tduy_dt,tduz_dt;

	// Get previous step
	tx0 = P2[i].x0;
	ty0 = P2[i].y0;
	tz0 = P2[i].z0;

	tux0=P2[i].ux0;
	tuy0=P2[i].uy0;
	tuz0=P2[i].uz0;

	//////// force to solid for no-slip boundary condition ////////
	// tdux_dt=P3[i].ftotalx-P1[i].fbx/P1[i].rho;
	// tduy_dt=P3[i].ftotaly-P1[i].fby/P1[i].rho;
	// tduz_dt=P3[i].ftotalz-P1[i].fbz/P1[i].rho;
	// two way coupling
	// tdux_dt=P1[i].ax-P1[i].fbx/P1[i].rho;
	// tduy_dt=P1[i].ay-P1[i].fby/P1[i].rho;
	// tduz_dt=P1[i].az-P1[i].fbz/P1[i].rho;
	// one way coupling
	tdux_dt=P1[i].ax;
	tduy_dt=P1[i].ay;
	tduz_dt=P1[i].az;

	// if(P1[i].ccc==3){
	// 	tux0=0.0;
	// 	tuy0=0.0;
	// 	tuz0=0.0;
	// 	tdux_dt=tduy_dt=tduz_dt=0.0;
	// }

	//update velocity

	tuxc = tux0 + tdux_dt*t_dt;
	tuyc = tuy0 + tduy_dt*t_dt;
	// tuzc = tuz0 + tduz_dt0*t_dt;

	// update position

	txc = tx0 + tuxc*(t_dt);
	tyc = ty0 + tuyc*(t_dt);
	// tzc = tz0 + tuzc*(t_dt);

	TP1[i].x = txc;
	TP1[i].y = tyc;
	// TP1[i].z = tzc;

	TP1[i].ux = tuxc;
	TP1[i].uy = tuyc;
	// TP1[i].uz = tuzc;

	// if(P1[i].p_type>=2000)	TP1[i].rho=P2[i].rho_ref/(P1[i].jacob+1e-20);

		


	TP1[i].fbx = P1[i].fbx;
	TP1[i].fby = P1[i].fby;
	TP1[i].fbz = P1[i].fbz;

	TP1[i].ux_i = P1[i].ux_i;
	TP1[i].uy_i = P1[i].uy_i;
	TP1[i].uz_i = P1[i].uz_i;

	
	TP1[i].ax = P1[i].ax;
	TP1[i].ay = P1[i].ay;
	TP1[i].az = P1[i].az;

	TP1[i].concentration=P1[i].concentration;

	TP1[i].vol=P1[i].vol;
	TP1[i].vol0=P1[i].vol0;

	//Structure modeling

	// TP1[i].F11=P1[i].F11;
	// TP1[i].F12=P1[i].F12;
	// TP1[i].F13=P1[i].F13;
	// TP1[i].F21=P1[i].F21;
	// TP1[i].F22=P1[i].F22;
	// TP1[i].F23=P1[i].F23;
	// TP1[i].F31=P1[i].F31;
	// TP1[i].F32=P1[i].F32;
	// TP1[i].F33=P1[i].F33;

	// TP1[i].px=P1[i].px;
	// TP1[i].py=P1[i].py;
	// TP1[i].pz=P1[i].pz;

	// TP1[i].jacob=P1[i].jacob;


}
