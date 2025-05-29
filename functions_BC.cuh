
////////////////////////////////////////////////////////////////////////
__host__ __device__ void penetration_box_2D(Real*xi_, Real*yi_, Real*uxi_, Real*uyi_)
{

	Real F_box;
	Real x_b_max,x_b_min,y_b_max,y_b_min;
	Real xi=*xi_;
	Real yi=*yi_;
	Real uxi=*uxi_;
	Real uyi=*uyi_;

	// x_b_max=k_x_max+margin;
	// x_b_min=k_x_min-margin;
	// y_b_max=k_y_max+margin;
	// y_b_min=k_y_min-margin;

	x_b_max=2.0196;
	x_b_min=-1.2;
	y_b_max=1.8;
	y_b_min=0.0;

	F_box=fmax(((xi-x_b_max)*(xi-x_b_min)),((yi-y_b_max)*(yi-y_b_min)));
	if(F_box<=0) return;

	Real cpx,cpy;
	Real sgn_x,sgn_y,sgn_m;
	Real nx_box,ny_box;

	//contact point
	cpx=fmin(x_b_max,fmax(x_b_min,xi));
	cpy=fmin(y_b_max,fmax(y_b_min,yi));
	sgn_x=(cpx-xi)/(abs(cpx-xi)+1e-10);
	sgn_y=(cpy-yi)/(abs(cpy-yi)+1e-10);
	sgn_m=sqrtf(sgn_x*sgn_x+sgn_y*sgn_y);
	nx_box=sgn_x/(sgn_m+1e-10);
	ny_box=sgn_y/(sgn_m+1e-10);

	*xi_=cpx;
	*yi_=cpy;
	*uxi_=uxi-2*(uxi*nx_box+uyi*ny_box)*nx_box;
	*uyi_=uyi-2*(uxi*nx_box+uyi*ny_box)*ny_box;
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_boundary2D(int_t*g_str,int_t*g_end,part1*P1)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>i_type_crt) return;
	

	int_t icell,jcell;
	uint_t p_type_i;
	p_type_i=P1[i].p_type;
	// if(p_type_i>0) return;

	Real xi,yi,uxi,uyi;
	Real search_range,tmp_h,tmp_A;
	Real tmpx,tmpy,flt;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	xi=P1[i].x;
	yi=P1[i].y;
	uxi=P1[i].ux;
	uyi=P1[i].uy;


	if((p_type_i>0)&(p_type_i!=MOVING)){
	}else{
		if(k_noslip_bc==1){
		// noslip boundary condition
			// calculate I,J,K in cell
			if((k_x_max==k_x_min)){icell=0;}
			else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
			if((k_y_max==k_y_min)){jcell=0;}
			else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
			// out-of-range handling
			if(icell<0) icell=0;	if(jcell<0) jcell=0;

			tmpx=0.0;
			tmpy=0.0;
			flt=1.0e-10;
			for(int_t y=-P1[i].ncell;y<=P1[i].ncell;y++){
				for(int_t x=-P1[i].ncell;x<=P1[i].ncell;x++){
					// int_t k=(icell+x)+k_NI*(jcell+y);
					int_t k=idx_cell(icell+x,jcell+y,0);

					if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))) continue;
					if(g_str[k]!=cu_memset){
						int_t fend=g_end[k];
						for(int_t j=g_str[k];j<fend;j++){
							Real xj,yj,tdist;
							xj=P1[j].x;
							yj=P1[j].y;

							tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj))+1e-20;
							Real tmp_hij;
							if(k_multi_type==3||k_multi_type==4)	tmp_hij=(P1[i].h+P1[j].h)/2;
							else	tmp_hij=P1[i].h;
							Real tmp_Aij=calc_tmpA(tmp_hij);

							search_range=k_search_kappa*tmp_hij;
							if(tdist<search_range){
								uint_t p_type_j;
								Real twij,uxj,uyj;
								twij=calc_kernel_wij(tmp_Aij,tmp_hij,tdist);
								p_type_j=P1[j].p_type;
								uxj=P1[j].ux;
								uyj=P1[j].uy;

								// if(p_type_j>0){
								if((p_type_j>0)&(p_type_j!=MOVING)){//YHS
									tmpx+=uxj*twij;
									tmpy+=uyj*twij;
									flt+=twij;
								}
							}
						}
					}
				}
			}
			if(k_noslip_bc==1){
				if (((p_type_i <= BOUNDARY) || (p_type_i==MOVING))){
					P1[i].ux=2*uxi*(p_type_i==MOVING)-tmpx/flt;
					P1[i].uy=2*uyi*(p_type_i==MOVING)-tmpy/flt;
				}
			}
		}
	}


		// penetration box
		if(k_penetration_solve==1&&P1[i].p_type>0) {
			penetration_box_2D(&xi, &yi, &uxi, &uyi);
			P1[i].x=xi;
			P1[i].y=yi;
			P1[i].ux=uxi;
			P1[i].uy=uyi;
		}
}

////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_boundary3D(int_t*g_str,int_t*g_end,part1*P1)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part3) return;
	if(P1[i].i_type>i_type_crt) return;

	int_t icell,jcell,kcell;
	uint_t p_type_i;
	p_type_i=P1[i].p_type;
	if(p_type_i>0) return;

	Real xi,yi,zi,uxi,uyi,uzi;
	Real search_range,tmp_h,tmp_A;
	Real tmpx,tmpy,tmpz,flt;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;
	uxi=P1[i].ux;
	uyi=P1[i].uy;
	uzi=P1[i].uz;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	if((k_z_max==k_z_min)){kcell=0;}
	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

	tmpx=0.0;
	tmpy=0.0;
	tmpz=0.0;
	flt=1.0e-10;
	for(int_t z=-P1[i].ncell;z<=P1[i].ncell;z++){
		for(int_t y=-P1[i].ncell;y<=P1[i].ncell;y++){
			for(int_t x=-P1[i].ncell;x<=P1[i].ncell;x++){
				int_t k=idx_cell(icell+x,jcell+y,kcell+z);

				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;
				if(g_str[k]!=cu_memset){
					int_t fend=g_end[k];
					for(int_t j=g_str[k];j<fend;j++){
						Real xj,yj,zj,tdist;
						xj=P1[j].x;
						yj=P1[j].y;
						zj=P1[j].z;

						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj))+1e-20;
						if(tdist<search_range){
							uint_t p_type_j;
							Real twij,uxj,uyj,uzj;
							twij=calc_kernel_wij(tmp_A,tmp_h,tdist);
							p_type_j=P1[j].p_type;
							uxj=P1[j].ux;
							uyj=P1[j].uy;
							uzj=P1[j].uz;

							// noslip boundary condition
							if(k_noslip_bc==1){
								if(p_type_j>0){
									tmpx+=uxj*twij;
									tmpy+=uyj*twij;
									tmpz+=uzj*twij;
									flt+=twij;
								}
							}


						}
					}
				}
			}
		}
	}

	// noslip boundary condition
	if(k_noslip_bc==1){
		if ((p_type_i == BOUNDARY) || (p_type_i==MOVING)){
			P1[i].ux=2*uxi*(p_type_i==MOVING)-tmpx/flt;
			P1[i].uy=2*uyi*(p_type_i==MOVING)-tmpy/flt;
			P1[i].uz=2*uzi*(p_type_i==MOVING)-tmpz/flt;

		}
	}
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_Neumann_boundary2D(int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if((P1[i].p_type>0)&&(P1[i].p_type!=MOVING))	return;		// Immersed Boundary Method

	int_t icell,jcell;
	Real xi, xgnode_i, yi,ygnode_i;
	Real uxi, uyi, rhoi, presi, tempi;
	Real rho_ref_i;
	Real search_range, tmp_h, tmp_A;
	Real tflt_s;
	Real t1, t2;
	Real tlambda;
	Real tpres, ttemp, thpresx, thpresy, tux, tuy, aix, aiy, aiz;
	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	// tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	xi=P1[i].x;
	yi=P1[i].y;
	uxi=P1[i].ux;
	uyi=P1[i].uy;
	rhoi=P1[i].rho;
	presi=P1[i].pres;
	tempi=P1[i].temp;

	tux=tuy=ttemp=tpres=thpresx=thpresy=tlambda=tflt_s=0.0;
	t1=t2=0.0;

	Real tt = 0.0;
	Real tD=0.0;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;

	for(int_t y=-P1[i].ncell;y<=P1[i].ncell;y++){
		for(int_t x=-P1[i].ncell;x<=P1[i].ncell;x++){
			// int_t k=(icell+x)+k_NI*(jcell+y);
			int_t k=idx_cell(icell+x,jcell+y,0);
			if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))) continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					Real xj,yj,tdist,uxj,uyj,mj,rhoj,presj,tempj,volj;
					Real lambdaj;

					xj=P1[j].x;
					yj=P1[j].y;
					mj=P1[j].m;
					rhoj=P1[j].rho;
					uxj=P1[j].ux;
					uyj=P1[j].uy;
					presj=P1[j].pres;
					tempj=P1[j].temp;
					volj=P1[j].vol;
					lambdaj=P3[j].lambda;

					tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj))+1e-20;
					if(P1[j].p_type<1000){
					Real tmp_hij;
					if(k_multi_type==3||k_multi_type==4)	tmp_hij=(P1[i].h+P1[j].h)/2;
					else	tmp_hij=P1[i].h;
					Real tmp_Aij=calc_tmpA(tmp_hij);

					search_range=k_search_kappa*tmp_hij;
					if(tdist<search_range){
					// if(tdist<search_range){
						Real twij, tdwij, tdwx, tdwy;
						int_t ptype_j=P1[j].p_type;
						int_t buffer_type_j=P1[j].buffer_type;
						twij=calc_kernel_wij(tmp_Aij,tmp_hij,tdist);
						tdwx=(xi-xj)/tdist * tdwij;
						tdwy=(yi-yj)/tdist * tdwij;
						tpres+=presj*twij*(ptype_j>0);
						thpresx+=rhoj*(xi-xj)*twij*(ptype_j>0);
						thpresy+=rhoj*(yi-yj)*twij*(ptype_j>0);
						ttemp+=tempj*twij*(ptype_j>0);
						tlambda+=lambdaj*twij*(ptype_j>0);
						tux+=uxj*twij*(ptype_j>0);
						tuy+=uyj*twij*(ptype_j>0);
						tflt_s+=twij*(ptype_j>0);
						t1+=twij*(ptype_j==1);
						t2+=twij*(ptype_j==2);

						tt = mj/rhoj;
						tD+=P1[j].D*twij*(ptype_j>0);

					}
				}
				}
			}
		}
	}


	if(tflt_s<1e-6){
		P1[i].rho=P2[i].rho_ref;
		P1[i].pres=0.0;
	}else{
	P1[i].pres = (tpres + (-0.0*thpresx + (-Gravitational_CONST-0.0)*thpresy))/tflt_s;
	P3[i].lambda = tlambda/tflt_s;
	P1[i].D = tD/tflt_s;
	
	// if(t2>1e-6&&t1<1e-6&&k_solver_type==Isph)		P1[i].pres = (0.0 + (-0.0*thpresx + (-Gravitational_CONST-0.0)*thpresy))/tflt_s;
	// if(yi>1.8)		P1[i].pres = (0.0 + (-0.0*thpresx + (-Gravitational_CONST-0.0)*thpresy))/tflt_s;

	// 압력이 무한히 올라가
	if(k_solver_type==Wcsph){
	Real rho0=fmax(1e-6,k_rho0_eos);
	double B = rho0*k_soundspeed*k_soundspeed/k_gamma;
	double K = P1[i].pres/B + 1.0;
	P1[i].rho = P2[i].rho_ref*pow(K, 1.0/k_gamma);
	}
	if(P1[i].p_type==0)	P1[i].temp = ttemp/tflt_s;
	}
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_Neumann_boundary3D(int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part3) return;
	if((P1[i].p_type>0)&&(P1[i].p_type!=MOVING))	return;		// Immersed Boundary Method


	int_t icell,jcell,kcell;
	Real xi, yi, zi;
	Real uxi, uyi, uzi;
	Real rhoi, presi;
	Real rho_ref_i;
	Real search_range, tmp_h, tmp_A;
	Real tflt_s;
	Real tux, tuy, tuz;
	Real tpres, thpres, ttemp, aix, aiy, aiz;
	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;
	uxi=P1[i].ux;
	uyi=P1[i].uy;
	uzi=P1[i].uz;
	rhoi=P1[i].rho;
	presi=P1[i].pres;

	tpres=thpres=ttemp=tflt_s=0.0;
	tux=tuy=tuz=0.0;

	// calculate I,J,K in cell
if((k_x_max==k_x_min)){icell=0;}
else{icell=min(floor((xi-k_x_min)/k_dcell),k_NI-1);}
if((k_y_max==k_y_min)){jcell=0;}
else{jcell=min(floor((yi-k_y_min)/k_dcell),k_NJ-1);}
if((k_z_max==k_z_min)){kcell=0;}
else{kcell=min(floor((zi-k_z_min)/k_dcell),k_NK-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

	for(int_t z=-P1[i].ncell;z<=P1[i].ncell;z++){
		for(int_t y=-P1[i].ncell;y<=P1[i].ncell;y++){
			for(int_t x=-P1[i].ncell;x<=P1[i].ncell;x++){
				// int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				int_t k=idx_cell(icell+x,jcell+y,kcell+z);
				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;
				if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					Real xj,yj,zj,uxj,uyj,uzj,tdist,mj,rhoj,presj,tempj;
					Real ajx,ajy,ajz;

					xj=P1[j].x;
					yj=P1[j].y;
					zj=P1[j].z;
					uxj=P1[j].ux;
					uyj=P1[j].uy;
					uzj=P1[j].uz;
					ajx=P3[j].ftotalx;
					ajy=P3[j].ftotaly;
					ajz=P3[j].ftotalz;
					mj=P1[j].m;
					rhoj=P1[j].rho;
					presj=P1[j].pres;
					tempj=P1[j].temp;

					tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj))+1e-20;
					if(P1[j].p_type<1000){

					if(tdist<search_range){
					// if(tdist<search_range){
						Real twij, tdwij, tdwx, tdwy;
						int_t ptype_j=P1[j].p_type;
						int_t buffer_type_j=P1[j].buffer_type;
						twij=calc_kernel_wij(tmp_A,tmp_h,tdist);
						tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);
						tdwx=(xi-xj)/tdist * tdwij;
						tdwy=(yi-yj)/tdist * tdwij;
						tdwy=(zi-zj)/tdist * tdwij;

						tpres+=mj/rhoj*presj*twij*((ptype_j>0)&&(ptype_j!=MOVING));
						ttemp+=mj/rhoj*tempj*twij*((ptype_j>0)&&(ptype_j!=MOVING));
						thpres+=((P3[j].ftotalx)*(xj-xi)+(P3[j].ftotaly)*(yj-yi)+(P3[j].ftotalz)*(zj-zi))*mj/rhoj*twij*((ptype_j>0)&&(ptype_j!=MOVING));
						tux+=mj/rhoj*uxj*twij*((ptype_j>0)&&(ptype_j!=MOVING));
						tuy+=mj/rhoj*uyj*twij*((ptype_j>0)&&(ptype_j!=MOVING));
						tuz+=mj/rhoj*uzj*twij*((ptype_j>0)&&(ptype_j!=MOVING));
						tflt_s+=mj/rhoj*twij*((ptype_j>0)&&(ptype_j!=MOVING));

					}
				}
			}
				}
			}
		}
	}


	if(tflt_s<1e-6){
		P1[i].rho=P2[i].rho_ref;
		P1[i].pres=0.0;
	}else{
	// P1[i].pres = (tpres-rhoi*thpres)/tflt_s;
	P1[i].pres = (tpres)/tflt_s;
	
	if(P1[i].p_type==0)	P1[i].temp = ttemp/tflt_s;
	if(k_solver_type==Wcsph){
	double B = 1000.0*k_soundspeed*k_soundspeed/k_gamma;
	double K = P1[i].pres/B + 1.0;
	P1[i].rho = P2[i].rho_ref*pow(K, 1.0/k_gamma);
	}

	}
// 	if(P1[i].pres<0.0){
// 		P1[i].pres=0.0;
// }
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_open_boundary_extrapolation2D(Real ttime, int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3,int_t tcount,Real tdt)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part3) return;
	if(P1[i].buffer_type<=0) return;
	// if((P1[i].elix<1e-10)&(P1[i].eliy<1e-10)) return;
	if(P1[i].p_type>=1000)	return;		// Immersed Boundary Method

	int_t icell,jcell;
	Real xi, xgnode_i, yi,ygnode_i;
	Real rho_ref_i;
	Real search_range, tmp_h, tmp_A;
	Real tflt_s;
	Real trho, ttemp, tpres, tux, tuy;
	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	xi=P1[i].x;
	yi=P1[i].y;

	// calculate ghost node

	if(P1[i].buffer_type==Inlet){
		xgnode_i=xi;
		ygnode_i=2*L2-yi;
	}else if(P1[i].buffer_type==Outlet){
		xgnode_i=xi;
		ygnode_i=2*L3-yi;
	}else if(P1[i].buffer_type==Left){
		xgnode_i=2*(L_left)-xi;
		ygnode_i=yi;
	}else if(P1[i].buffer_type==Right){
		xgnode_i=2*(L_right)-xi;
		ygnode_i=yi;
	}

	trho=tux=tuy=tflt_s=tpres=0.0;

	Real tD=0.0;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xgnode_i-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((ygnode_i-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;

	for(int_t y=-P1[i].ncell;y<=P1[i].ncell;y++){
		for(int_t x=-P1[i].ncell;x<=P1[i].ncell;x++){
			int_t k=idx_cell(icell+x,jcell+y,0);
			if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))) continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					Real xj,yj,tdist,uxj,uyj,mj,rhoj,presj;
					Real volj;

					xj=P1[j].x;
					yj=P1[j].y;
					mj=P1[j].m;
					rhoj=P1[j].rho;
					volj=P1[j].vol;
					uxj=P1[j].ux;
					uyj=P1[j].uy;
					presj=P1[j].pres;

					tdist=sqrt((xgnode_i-xj)*(xgnode_i-xj)+(ygnode_i-yj)*(ygnode_i-yj))+1e-20;
					if(P1[j].p_type<1000){

					if(tdist<search_range){
						Real twij, tdwij, tdwx, tdwy;
						int_t ptype_j=P1[j].p_type;
						int_t buffer_type_j=P1[j].buffer_type;
						twij=calc_kernel_wij(tmp_A,tmp_h,tdist);
						tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);
						tdwx=(xgnode_i-xj)/tdist * tdwij;
						tdwy=(ygnode_i-yj)/tdist * tdwij;

						tux+=uxj*twij*mj/rhoj*(ptype_j>0);
						tuy+=uyj*twij*mj/rhoj*(ptype_j>0);
						trho+=rhoj*twij*mj/rhoj*(ptype_j>0);
						tpres+presj*twij*mj/rhoj*(ptype_j>0);
						tflt_s+=twij*mj/rhoj*(ptype_j>0);
						tD+=P1[j].D*twij*mj/rhoj*(ptype_j>0);

					}
				}
				}
			}
		}
	}

		if (k_open_boundary==1){

				if(P1[i].buffer_type!=Outlet){

					if(P1[i].buffer_type==Left || P1[i].buffer_type==Right){
						P1[i].ux=tux/tflt_s;
						P1[i].rho=trho/tflt_s;
						Real tB,tP,rho0,tmpres;
						rho0=fmax(1e-6,k_rho0_eos);
						tB=k_soundspeed*k_soundspeed*rho0/k_gamma;
						tP=pow(P1[i].rho/P2[i].rho_ref,k_gamma);
						tmpres=tB*(tP-1.0);
						P1[i].pres=tmpres;
						if((P1[i].elix<0.9999)&&(P1[i].p_type>0))	P1[i].m=P1[i].rho*P1[i].vol;

					}

					if(P1[i].buffer_type==Inlet){
						
						P1[i].rho=trho/tflt_s;
						Real tB,tP,rho0,tmpres;
						rho0=fmax(1e-6,k_rho0_eos);
						tB=k_soundspeed*k_soundspeed*rho0/k_gamma;
						tP=pow(P1[i].rho/P2[i].rho_ref,k_gamma);
						tmpres=tB*(tP-1.0);
						P1[i].pres=tmpres;
						P1[i].ux = Inlet_Velocity;
						if((P1[i].elix<0.9999)&&(P1[i].p_type>0))	P1[i].m=P1[i].rho*P1[i].vol;
					}
				}
			}
		if (k_open_boundary==2){
			if(P1[i].buffer_type==Inlet){
				P1[i].pres=tpres/tflt_s;
				// P1[i].rho=trho/tflt_s;
				P1[i].flt_s=tflt_s;
				P1[i].ux=tux/tflt_s;
				P1[i].uy=tuy/tflt_s;
				P1[i].D=tD/tflt_s;
			}
			if(P1[i].buffer_type==Outlet){
				P1[i].pres=tpres/tflt_s;
				P1[i].ux=tux/tflt_s;
				P1[i].uy=tuy/tflt_s;
				P1[i].D=tD/tflt_s;
				// P1[i].rho=trho/tflt_s;
			}
	}



}
////////////////////////////////////////////////////////////////////////
__global__ void set_inlet_buffer(part1*P1, Real ttime)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part3) return;
	if(P1[i].buffer_type!=1) return;

	Real xi = 0;
	Real H = 0.2;
	Real d = 0.3;
	Real C = sqrt(Gravitational_CONST*(H+d));
	Real T = 2*d/C*sqrt(4*d/3/H)*(3.8+H/d);
	// Real T=0;
	Real X = sqrt(3*H/4/d/d/d)*(xi-C*(ttime-T));

	// Real limit = H*(1.0/(exp(2*X)+exp(-2*X)+2.0))+d;
	Real k, w;
	k = PI/1.0;
	w = PI/1.0;

	Real limit = H/cos(PI/2.0-w*ttime)+d;


	if(P1[i].y>limit){
		P1[i].i_type=3;
	}else{
		P1[i].i_type=2;
	}

}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_periodic_boundary_extrapolation2D(Real ttime, int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3,int_t tcount,Real tdt)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part3) return;
	if(P1[i].buffer_type<=0) return;
	// if((P1[i].elix<1e-10)&(P1[i].eliy<1e-10)) return;
	if(P1[i].p_type>=1000)	return;		// Immersed Boundary Method

	int_t icell,jcell;
	Real xi, xgnode_i, yi,ygnode_i;
	Real rho_ref_i;
	Real search_range, tmp_h, tmp_A, trho, tdrho_x, tdrho_y, tux, tdux_x, tdux_y, tuy, tduy_x, tduy_y;
	Real drho_x, drho_y;
	Real dux_x, dux_y, duy_x, duy_y;
	Real tflt_s;
	Real tpres, tdpres_x, tdpres_y, dpres_x, dpres_y;
	tmp_h=P1[i].h;
	// tmp_A=calc_tmpA(Extrapolation_Length*tmp_h);
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	Real p_ref, ux_ref, uy_ref, rho_ref;
	Real J1, J2, J3, J4;

	xi=P1[i].x;
	yi=P1[i].y;

	// calculate ghost node

	if(P1[i].buffer_type==Inlet){
		xgnode_i=xi;
		ygnode_i=2*L2-yi;
	}else if(P1[i].buffer_type==Outlet){
		xgnode_i=xi;
		ygnode_i=2*L3-yi;
	}else if(P1[i].buffer_type==Left){
		xgnode_i=2*(L_left)-xi;
		ygnode_i=yi;
	}else if(P1[i].buffer_type==Right){
		xgnode_i=2*(L_right)-xi;
		ygnode_i=yi;
	}
	if(P1[i].buffer_type==5){
		xgnode_i=2*(L_left)-xi;
		ygnode_i=2*L3-yi;
	}else if(P1[i].buffer_type==6){
		xgnode_i=2*(L_right)-xi;
		ygnode_i=2*L3-yi;
	}else if(P1[i].buffer_type==7){
		xgnode_i=2*(L_left)-xi;
		ygnode_i=2*L2-yi;
	}else if(P1[i].buffer_type==8){
		xgnode_i=2*(L_right)-xi;
		ygnode_i=2*L2-yi;
	}

	trho=tdrho_x=tdrho_y=tux=tdux_x=tdux_y=tuy=tduy_x=tduy_y=tflt_s=tpres=tdpres_x=tdpres_y=0.0;
	p_ref=0.0; uy_ref=Inlet_Velocity;  ux_ref=0.0; rho_ref=Inlet_Density;

	J1=J2=J3=J4=0.0; // characteristic wave


	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xgnode_i-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((ygnode_i-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;

	for(int_t y=-P1[i].ncell;y<=P1[i].ncell;y++){
		for(int_t x=-P1[i].ncell;x<=P1[i].ncell;x++){
			// int_t k=(icell+x)+k_NI*(jcell+y);
			int_t k=idx_cell(icell+x,jcell+y,0);
			if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))) continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					Real xj,yj,tdist,uxj,uyj,mj,rhoj,presj;
					Real volj;

					xj=P1[j].x;
					yj=P1[j].y;
					mj=P1[j].m;
					rhoj=P1[j].rho;
					volj=P1[j].vol;
					uxj=P1[j].ux;
					uyj=P1[j].uy;
					presj=P1[j].pres;

					tdist=sqrt((xgnode_i-xj)*(xgnode_i-xj)+(ygnode_i-yj)*(ygnode_i-yj))+1e-20;
					if(P1[j].p_type<1000){

						Real tmp_hij;
if(k_multi_type==3||k_multi_type==4)	tmp_hij=(P1[i].h+P1[j].h)/2;
else	tmp_hij=P1[i].h;
						Real tmp_Aij=calc_tmpA(tmp_hij);
	
						search_range=k_search_kappa*fmax(P1[i].h,P1[j].h);
					if(tdist<search_range){
						Real twij, tdwij, tdwx, tdwy;
						int_t ptype_j=P1[j].p_type;
						int_t buffer_type_j=P1[j].buffer_type;
						// twij=calc_kernel_wij(tmp_A,Extrapolation_Length*tmp_h,tdist);
						// tdwij=calc_kernel_dwij(tmp_A,Extrapolation_Length*tmp_h,tdist);
						twij=calc_kernel_wij(tmp_Aij,tmp_hij,tdist);
						tdwij=calc_kernel_dwij(tmp_Aij,tmp_hij,tdist);
						tdwx=(xgnode_i-xj)/tdist * tdwij;
						tdwy=(ygnode_i-yj)/tdist * tdwij;

						tux+=uxj*twij*mj/rhoj*(ptype_j>0)*(buffer_type_j==0);
						tdux_x+=uxj*tdwx*mj/rhoj*(ptype_j>0)*(buffer_type_j==0);
						tdux_y+=uxj*tdwy*mj/rhoj*(ptype_j>0)*(buffer_type_j==0);


						tuy+=uyj*twij*mj/rhoj*(ptype_j>0)*(buffer_type_j==0);
						tduy_x+=uyj*tdwx*mj/rhoj*(ptype_j>0)*(buffer_type_j==0);
						tduy_y+=uyj*tdwy*mj/rhoj*(ptype_j>0)*(buffer_type_j==0);

						trho+=rhoj*twij*mj/rhoj*(ptype_j>0)*(buffer_type_j==0);
						tdrho_x+=rhoj*tdwx*mj/rhoj*(ptype_j>0)*(buffer_type_j==0);
						tdrho_y+=rhoj*tdwy*mj/rhoj*(ptype_j>0)*(buffer_type_j==0);

						tpres+=mj/rhoj*presj*twij*(ptype_j>0)*(buffer_type_j==0);
						tdpres_x+=presj*tdwx*mj/rhoj*(ptype_j>0)*(buffer_type_j==0);
						tdpres_y+=presj*tdwy*mj/rhoj*(ptype_j>0)*(buffer_type_j==0);

						tflt_s+=mj/rhoj*twij*(ptype_j>0)*(buffer_type_j==0);

					}
				}
				}
			}
		}
	}

	if (k_open_boundary==2){

			if(k_solver_type==Wcsph){
			P1[i].pres=-tpres/tflt_s;
			Real rho0=fmax(1e-6,k_rho0_eos);
			Real rho_ref = P2[i].rho_ref;
			Real tB=k_soundspeed*k_soundspeed*rho0/k_gamma;
			P1[i].rho = rho_ref*pow(((P1[i].pres)/tB+1.0),(1.0/k_gamma));
			P1[i].flt_s=tflt_s;
			}else{
			P1[i].pres=-tpres/tflt_s;
			}

		}



}