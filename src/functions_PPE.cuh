__device__ Real calc_kernel_wij_ipf(Real tH,Real rr){

	Real tR,wij_ipf,tA;
	tR=wij_ipf=0.0;
	tA=1.0;
	Real eps;
	eps=tH/3.5;

	if(k_IPF_kernel_type==Cosine){
		if(k_kappa==1){
		tR=rr/tH;
		wij_ipf=-(tR<1)*tA*cos(3*PI/2*tR);}
		else	if(k_kappa==2){
		tR=rr/tH;
		wij_ipf=-(tR<2)*tA*cos(3*PI/4*tR);}
	}else	if(k_IPF_kernel_type==Gauss){
		tR=rr/tH;
		wij_ipf=(tR<1)*tA*exp(-pow(rr,2)/2/eps/eps);
	}else	if(k_IPF_kernel_type==Modified_Gaussian){
		tR=rr/tH;
		wij_ipf=(tR<1)*tA*rr*exp(-pow(rr,2)/2/eps/eps);
	}else	if(k_IPF_kernel_type==Cubic){
		tR=rr/tH;
		if(0<=tR&&tR<1/3)  wij_ipf=(tR<1)*tA*((3-3*tR)*(3-3*tR)*(3-3*tR)*(3-3*tR)*(3-3*tR)-6*(2-3*tR)*(2-3*tR)*(2-3*tR)*(2-3*tR)*(2-3*tR)+15*(1-3*tR)*(1-3*tR)*(1-3*tR)*(1-3*tR)*(1-3*tR));
		if(1/3<=tR&&tR<2/3)  wij_ipf=(tR<1)*tA*((3-3*tR)*(3-3*tR)*(3-3*tR)*(3-3*tR)*(3-3*tR)-6*(2-3*tR)*(2-3*tR)*(2-3*tR)*(2-3*tR)*(2-3*tR));
		if(2/3<=tR&&tR<1)  wij_ipf=(tR<1)*tA*((3-3*tR)*(3-3*tR)*(3-3*tR)*(3-3*tR)*(3-3*tR));
	}else	if(k_IPF_kernel_type==Wend2){
		tR=rr/tH*0.5;
		wij_ipf=(tR<1)*tA*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1+4*tR);
	}
	return wij_ipf;
}

__device__ Real calc_kernel_wij_half(Real tH,Real rr){

	Real tR,wij_half,tA;
	tR=wij_half=0.0;
	tA=1.0;
	Real eps0;
	eps0=tH/3.5*0.4;

	if(k_IPF_kernel_type==Cosine){
		tR=rr/tH;
		wij_half=0.0;
	}else	if(k_IPF_kernel_type==Gauss){
		tR=rr/tH;
		wij_half=(tR<1)*tA*exp(-pow(rr,2)/2/eps0/eps0);
	}else	if(k_IPF_kernel_type==Modified_Gaussian){
		tR=rr/tH;
		wij_half=(tR<1)*tA*rr*exp(-pow(rr,2)/2/eps0/eps0);
	}else	if(k_IPF_kernel_type==Cubic){
		tR=rr/tH;
		if(0<=(tR*2)&&(tR*2)<1/3)  wij_half=((tR*2)<1)*tA*((3-3*(tR*2))*(3-3*(tR*2))*(3-3*(tR*2))*(3-3*(tR*2))*(3-3*(tR*2))-6*(2-3*(tR*2))*(2-3*(tR*2))*(2-3*(tR*2))*(2-3*(tR*2))*(2-3*(tR*2))+15*(1-3*(tR*2))*(1-3*(tR*2))*(1-3*(tR*2))*(1-3*(tR*2))*(1-3*(tR*2)));
		if(1/3<=(tR*2)&&(tR*2)<2/3)  wij_half=((tR*2)<1)*tA*((3-3*(tR*2))*(3-3*(tR*2))*(3-3*(tR*2))*(3-3*(tR*2))*(3-3*(tR*2))-6*(2-3*(tR*2))*(2-3*(tR*2))*(2-3*(tR*2))*(2-3*(tR*2))*(2-3*(tR*2)));
		if(2/3<=(tR*2)&&(tR*2)<1)  wij_half=((tR*2)<1)*tA*((3-3*(tR*2))*(3-3*(tR*2))*(3-3*(tR*2))*(3-3*(tR*2))*(3-3*(tR*2)));
	}else	if(k_IPF_kernel_type==Wend2){
		tR=rr/tH*0.5;
		wij_half=((tR*2)<1)*tA*(1-(tR*2))*(1-(tR*2))*(1-(tR*2))*(1-(tR*2))*(1+4*(tR*2));
	}
	return wij_half;
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_PPE2D(Real tdt, int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3,int it)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>i_type_crt) return;
	// if(k_open_boundary>0 && P1[i].buffer_type>0) return;
	if(P1[i].p_type>=1000)	return;		// Immersed Boundary Method
	if(P1[i].p_type<=0)	return;		// Immersed Boundary Method
	if((k_solver_type==Icsph)&&(P1[i].p_type==2))	return;		// Immersed Boundary Method
	
	int_t icell,jcell;
	Real xi,yi,uxi,uyi,rhoi,pi;
	Real xistar,yistar,rhostar,rhoavg;
	Real search_range,hi,tmp_A;
	Real tmpx,tmpy,flt;
	Real bi, bix, biy;
	Real Aij;
	Real AijPij;

	hi=P1[i].h;
	tmp_A=calc_tmpA(hi);
	search_range=k_search_kappa*hi;	// search range

	xi=P1[i].x;
	yi=P1[i].y;
	xistar=P1[i].x_star;
	yistar=P1[i].y_star;
	uxi=P1[i].ux;
	uyi=P1[i].uy;
	rhoi=P1[i].rho;
	pi=P1[i].pres;

	rhostar=rhoavg=0.0;
	bi = Aij = AijPij = 0.0;
	bix = 0.0;
	biy = 0.0;
	flt = 0.0;

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
					Real xj,yj,uxj,uyj,tdist;
					Real xjstar,yjstar,tdiststar;
					Real rhoj,mj,pj,rho_ref_j;
					Real volj;
					int p_type_j;
					int itype;

					itype=P1[j].i_type;
					mj=P1[j].m;

						if(itype!=4){
					xj=P1[j].x;
					yj=P1[j].y;
					xjstar=P1[j].x_star;
					yjstar=P1[j].y_star;
					uxj=P1[j].ux;
					uyj=P1[j].uy;
					rhoj=P1[j].rho;
					rho_ref_j=P2[j].rho_ref;
					volj=P1[j].vol;
					pj=P1[j].pres;
					if(P1[j].p_type<=0)	pj += rhoi*P1[j].dpres;
				    if(P3[j].lambda<0.6 && P1[j].p_type==1) pj *=0.;

					tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj))+1e-20;
					tdiststar=sqrt((xistar-xjstar)*(xistar-xjstar)+(yistar-yjstar)*(yistar-yjstar))+1e-20;
					Real tmp_hij;
					if(k_multi_type==3||k_multi_type==4)	tmp_hij=(P1[i].h+P1[j].h)/2;
					else	tmp_hij=P1[i].h;
					Real tmp_Aij=calc_tmpA(tmp_hij);

					search_range=k_search_kappa*fmax(P1[i].h,P1[j].h);
					if(tdist<search_range){
						Real twij=calc_kernel_wij(tmp_Aij,tmp_hij,tdist);
						Real twijstar=calc_kernel_wij(tmp_Aij,tmp_hij,tdiststar);
						Real tdwij=calc_kernel_dwij(tmp_Aij,tmp_hij,tdist);

						Real tdwx=tdwij*(xi-xj)/tdist;
						Real tdwy=tdwij*(yi-yj)/tdist;

						Real tdwxc=tdwx*P3[i].Cm[0][0]+tdwy*P3[i].Cm[0][1];
						Real tdwyc=tdwx*P3[i].Cm[1][0]+tdwy*P3[i].Cm[1][1];

						bix += mj/rhoj*(uxj-uxi)*tdwx*(2.0*rhoi*rhoj/(rhoi+rhoj));
						biy += mj/rhoj*(uyj-uyi)*tdwy*(2.0*rhoi*rhoj/(rhoi+rhoj));

						rhostar += (mj/rhoj)*twijstar*(2.0*rhoi*rhoj/(rhoi+rhoj));
						rhoavg += (mj/rhoj)*twij*(2.0*rhoi*rhoj/(rhoi+rhoj));

						flt += (mj/rhoj)*twij;

						Real c = 2.0*mj/rhoj*((xi-xj)*tdwx+(yi-yj)*tdwy)/tdist/tdist;

						Aij += c;
						AijPij += c*(pj);

					}
				}
			}
				}
		}
	}
	bi = (bix+biy)/tdt;

	P1[i].PPE1 = (bix+biy)/tdt;
	P1[i].PPE2 = (rhoi-rhostar/flt)/tdt/tdt;

	P1[i].pres = (bi+AijPij)/Aij;
	// if(P3[i].lambda < 0.6) P1[i].pres = 0.;
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_advection_force2D(Real ttime,int_t inout,int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part3) return;
	if(!k_open_boundary && P1[i].i_type!=inout) return;
	if(k_open_boundary>0 && P1[i].buffer_type>0) return;
	if(P1[i].p_type>=1000)	return;	
	if(P1[i].p_type<=0)	return;	

	int_t ptypei;
	int_t icell,jcell;
	Real xi,yi,uxi,uyi,kci,eta;
	Real pi,hi,mi,mi8,mri,rhoi,tempi,visi,betai;
	Real diffi,concni;
	Real nxi,nyi,nmagi,sigmai;			// for surface tension
	Real nx_ci,ny_ci,nmag_ci,curvi; 	// for surface tension
	Real search_range,tmp_A,tmp_Rc,tmp_Rd;
	Real tmpx,tmpy,tmpn,tmpd;
	Real tmp_fsn, tmp_fsd;
	Real eulerx, eulery, eulert;
	Real cpi, temp0;
	Real Di = P1[i].D;

	ptypei=P1[i].p_type;

	xi=P1[i].x;
	yi=P1[i].y;
	uxi=P1[i].ux;
	uyi=P1[i].uy;
	hi=P1[i].h;
	tempi=P1[i].temp;
	mi=P1[i].m;
	rhoi=P1[i].rho;
	temp0=P1[i].temp0;

	tmp_A=calc_tmpA(hi);
	search_range=k_search_kappa*hi;	// search range
	mi8=0.08/mi;
	mri=(mi/rhoi);

	visi=viscosity(tempi,Di,ptypei)+P3[i].vis_t;
	betai=thermal_expansion(tempi,ptypei);

	if((k_fs_solve)&(k_surf_model==2)){

		nxi=P3[i].nx;
		nyi=P3[i].ny;
		nmagi=P3[i].nmag;
		nx_ci=P3[i].nx_c;
		ny_ci=P3[i].ny_c;
		nmag_ci=P3[i].nmag_c;

		sigmai=sigma(tempi,ptypei);
	}

	if(k_con_solve){
		kci=conductivity(tempi,ptypei);
		cpi=specific_heat(tempi,ptypei);
		eta=0.001*hi;
	}

	if(k_concn_solve){
		concni=P1[i].concn;
		diffi=diffusion_coefficient(tempi,ptypei);
	}


	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;

	tmpx=tmpy=0.0;
	tmpn=0.0;
	tmpd=1.0;
	tmp_Rc=0.0;
	tmp_Rd=0.0;
	tmp_fsn=0.0;
	tmp_fsd=0.0;
	eulerx=eulery=eulert=0.0;

	for(int_t y=-P1[i].ncell;y<=P1[i].ncell;y++){
		for(int_t x=-P1[i].ncell;x<=P1[i].ncell;x++){
			int_t k=idx_cell(icell+x,jcell+y,0);

			if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))) continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					Real xj,yj,tdist; //rr
					int itype;
					xj=P1[j].x;
					yj=P1[j].y;
					itype=P1[j].i_type;

					if(P1[j].p_type<1000){
						if(itype!=4){
					tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj))+1e-20;
					Real tmp_hij;
					if(k_multi_type==3||k_multi_type==4)	tmp_hij=(P1[i].h+P1[j].h)/2;
					else	tmp_hij=P1[i].h;
					Real tmp_Aij=calc_tmpA(tmp_hij);

					search_range=k_search_kappa*fmax(P1[i].h,P1[j].h);
					if(tdist<search_range){
						int_t ptypej;
						Real tdwx,tdwy,uxj,uyj,mj,tempj,rhoj,pj,hj,kcj,sum_con_H, diffj, concnj,tmprd;
						Real nx_cj,ny_cj,nmag_cj,Phi_s,tmpnt;	// for surface tension
						Real tdwxc, tdwyc;

						Real twij=calc_kernel_wij(tmp_Aij,tmp_hij,tdist);
						Real tdwij=calc_kernel_dwij(tmp_Aij,tmp_hij,tdist);

						Real wij_ipf=calc_kernel_wij_ipf(hi,tdist);
						Real wij_half=calc_kernel_wij_half(hi,tdist);


						tdwx=tdwij*(xi-xj)/tdist;
						tdwy=tdwij*(yi-yj)/tdist;

						tdwxc=tdwx*P3[i].Cm[0][0]+tdwy*P3[i].Cm[0][1];
						tdwyc=tdwx*P3[i].Cm[1][0]+tdwy*P3[i].Cm[1][1];

						ptypej=P1[j].p_type;
						uxj=P1[j].ux;
						uyj=P1[j].uy;
						mj=P1[j].m;
						tempj=P1[j].temp;
						rhoj=P1[j].rho;
						hj=P1[j].h;
						Real Dj=P1[j].D;

						if(k_fv_solve>0){
							if(k_fv_solve==Morris){
								Real C_v;
								Real visj=viscosity(tempj,Dj,ptypej)+P3[j].vis_t;
								C_v=(xi-xj)*tdwx+(yi-yj)*tdwy;
								C_v*=4*(mj/(rhoi*rhoj));
								C_v*=(visi*visj)/(visi+visj+1e-20);
								C_v/=tdist;
								C_v/=tdist;

								if(P1[i].p_type==0)	C_v=0.0;
								if(P1[j].p_type==0)	C_v=0.0;

								tmpx+=C_v*(uxi-uxj);
								tmpy+=C_v*(uyi-uyj);

								// Real visj,C_v;
								// visj=viscosity(tempj,ptypej)+P3[j].vis_t;
								// Real ex, ey;
								// ex = (xi-xj)/tdist;
								// ey = (yi-yj)/tdist;

								// C_v=2*visj/rhoj;
								// C_v*=mj/rhoj;
								// C_v*=(P3[i].L11*ex*tdwx+P3[i].L12*ey*tdwx+P3[i].L21*ex*tdwy+P3[i].L22*ey*tdwy);
						
								// if(P1[i].p_type==0)	C_v=0.0;
								// if(P1[j].p_type==0)	C_v=0.0;

								// tmpx+=C_v*((uxi-uxj)/tdist-(ex*P3[i].Gxx+ey*P3[i].Gxy));
								// tmpy+=C_v*((uyi-uyj)/tdist-(ex*P3[i].Gyx+ey*P3[i].Gyy));
	

							}else if(k_fv_solve==Monaghan){

								Real uij_xij=(uxi-uxj)*(xi-xj)+(uyi-uyj)*(yi-yj);
								Real h_ij,phi_ij,P_ij,P2_ij;
								Real visj=viscosity(tempj,Dj,ptypej)+P3[j].vis_t;
								// Monaghan type
								phi_ij=uij_xij;
								phi_ij/=(tdist*tdist+0.01*hi*hi);
								P_ij=-2.0*(k_dim+2.0)*phi_ij*mj/rhoj;
								Real nui,C_v;
								C_v=2.0*(visi*visj)/(visi+visj+1e-20);
								nui = C_v/rhoi;
								P_ij*= nui;

								if(P1[i].p_type==0)	P_ij=0.0;
								if(P1[j].p_type==0)	P_ij=0.0;

								tmpx+=-(P_ij)*tdwx;
								tmpy+=-(P_ij)*tdwy;


							}else if(k_fv_solve==Violeau){

								Real uij_xij=(uxi-uxj)*(xi-xj)+(uyi-uyj)*(yi-yj);
								Real h_ij,phi_ij,P_ij,P2_ij;
								Real visj=viscosity(tempj,Dj,ptypej)+P3[j].vis_t;
								// Violeau type
								phi_ij=uij_xij;
								phi_ij/=(tdist*tdist+0.01*hi*hi);
								P_ij=-(k_dim+2.0)*phi_ij*mj/rhoj;
								Real nui,C_v;
								C_v=2.0*(visi*visj)/(visi+visj+1e-20);
								nui = C_v/rhoi;
								P_ij*= nui;

								Real tmpv;
								tmpv=(mj/rhoj)*tdwij;
								tmpv*=nui;
								tmpv/=(tdist+0.01*hi*hi);

								tmpx+=-((P_ij)*tdwx-tmpv*(uxi-uxj));
								tmpy+=-((P_ij)*tdwy-tmpv*(uyi-uyj));
							}
						}
						// if(k_interface_solve){
						// 	int_t flag;
						// 	Real mrj,C_i;
						// 	flag=(ptypei!=BOUNDARY)&&(ptypei!=MOVING)&&(ptypej!=BOUNDARY)&&(ptypej!=MOVING)&&(ptypei!=ptypej);
						// 	mrj=mj/rhoj;
						// 	// Real C_p=-2.0*mj/(rhoi*rhoj)*(pi*rhoj+pj*rhoi)/(rhoi+rhoj);
						// 	// C_i=(abs(pi)*mri*mri+abs(pj)*mrj*mrj)*(flag);
						// 	C_i=2.0*0.01*mj/(rhoi*rhoj)*(abs(pi)*rhoj+abs(pj)*rhoi)/(rhoi+rhoj)*(flag);
						// 	tmpx+=C_i*tdwx;
						// 	tmpy+=C_i*tdwy;

						// }
						if(k_fb_solve){
							if((ptypei==FLUID)&(ptypej!=FLUID)){
								Real fb_ij=k_c_repulsive/(tdist+1e-10)/(tdist+1e-10)*twij*(2*mj/(mi+mj));
								tmpx+=fb_ij*(xi-xj);
								tmpy+=fb_ij*(yi-yj);
							}
						}
						if(k_fs_solve){
							if(k_surf_model==2){

							nx_cj=P3[j].nx_c;
							ny_cj=P3[j].ny_c;

							nmag_cj=P3[j].nmag_c;
							Phi_s=-(ptypei!= ptypej)+(ptypei==ptypej);

							tmpnt=((nx_ci/nmag_ci)-Phi_s*(nx_cj/nmag_cj))*(xj-xi);
							tmpnt+=((ny_ci/nmag_ci)-Phi_s*(ny_cj/nmag_cj))*(yj-yi);

							tmpnt*=k_dim*(mj/rhoj)*tdwij/tdist;
							tmp_fsn+=tmpnt;
							tmp_fsd+=(mj/rhoj)*tdist*abs(tdwij);
						}
						else	if(k_surf_model==1){
							Real Cs;
							if ((ptypei == 1) & (ptypej == 1))
							{
								Cs=s_ff1 * (A_ff1 * wij_half/(mi*(tdist + 1.0e-10)));
								Cs-=s_ff1 * (wij_ipf/(mi*(tdist + 1.0e-10)));
							}
							else if ((ptypei == 2) & (ptypej == 2))
							{
								Cs=s_ff2 * (A_ff2 * wij_half/(mi*(tdist + 1.0e-10)));
								Cs-=s_ff2 * (wij_ipf/(mi*(tdist + 1.0e-10)));
							}
							else if (((ptypei == 1) & (ptypej == 2)) || ((ptypei == 2) & (ptypej == 1)))
							{
								Cs=s_f1f2 * (A_f1f2 * wij_half/(mi*(tdist + 1.0e-10)));
								Cs-=s_f1f2 * (wij_ipf/(mi*(tdist + 1.0e-10)));
							}
							else if (((ptypei == 0) & (ptypej == 1)) || ((ptypei == 1) & (ptypej == 0)))
							{
								Cs=s_sf1 * (A_sf1 * wij_half/(mi*(tdist + 1.0e-10)));
								Cs-=s_sf1 * (wij_ipf/(mi*(tdist + 1.0e-10)));
							}
							else if (((ptypei == 0) & (ptypej == 2)) || ((ptypei == 2) & (ptypej == 0)))
							{
								Cs=s_sf2 * (A_sf2 * wij_half/(mi*(tdist + 1.0e-10)));
								Cs-=s_sf2 * (wij_ipf/(mi*(tdist + 1.0e-10)));
							}
							else if (((ptypei == 9) & (ptypej == 1)) || ((ptypei == 1) & (ptypej == 9)))
							{
								Cs=s_s2f1 * (A_s2f1 * wij_half/(mi*(tdist + 1.0e-10)));
								Cs-=s_s2f1 * (wij_ipf/(mi*(tdist + 1.0e-10)));
							}
							else if (((ptypei == 9) & (ptypej == 2)) || ((ptypei == 2) & (ptypej == 9)))
							{
								Cs=s_s2f2 * (A_s2f2 * wij_half/(mi*(tdist + 1.0e-10)));
								Cs-=s_s2f2 * (wij_ipf/(mi*(tdist + 1.0e-10)));
							}
							else
							{
								Cs=0.0;
							}
							tmpx+= Cs*(xi - xj);
							tmpy+= Cs*(yi - yj);
						}
					}
					// if((P1[i].elix<1.0)||(P1[i].eliy<1.0)){
						eulerx += ((uxi*(uxj-uxi)*tdwx+uyi*(uxj-uxi)*tdwy)*mj/rhoj*(1-P1[i].elix)-(P1[i].dux*(uxj-uxi)*tdwx+P1[i].duy*(uxj-uxi)*tdwy)*mj/rhoj)*(P1[i].p_type==P1[j].p_type);
						eulery += ((uxi*(uyj-uyi)*tdwx+uyi*(uyj-uyi)*tdwy)*mj/rhoj*(1-P1[i].eliy)-(P1[i].dux*(uyj-uyi)*tdwx+P1[i].duy*(uyj-uyi)*tdwy)*mj/rhoj)*(P1[i].p_type==P1[j].p_type);
						if(k_con_solve){
							eulert += (uxi*(tempj-tempi)*tdwx+uyi*(tempj-tempi)*tdwy)*mj/rhoj*(1-P1[i].elix);
						}
					// }
						if(k_con_solve){
							kcj=conductivity(tempj, ptypej);
							sum_con_H=4.0*mj*kcj*kci*(tempi-tempj)*tdwij;
							sum_con_H/=(tdist+1e-10)*rhoi*rhoj*(kci+kcj);
							tmp_Rc+=sum_con_H;
						}
						if(k_concn_solve){
							concnj=P1[j].concn;
							diffj=diffusion_coefficient(tempj,ptypej);
							tmprd=mi*(diffi*rhoi+diffj*rhoj);
							tmprd*=((xi-xj)*tdwx+(yi-yj)*tdwy);
							tmprd/=(rhoi*rhoj*(tdist*tdist+0.01*hi*hi));
							tmprd*=(concni-concnj)*(ptypei==ptypej);

							tmp_Rd+=tmprd;
						}
					}
				}
				}
			}
			}
		}
	}
	if(k_fg_solve) tmpy+=-Gravitational_CONST;
	if((k_boussinesq_solve)&(ptypei>0)&(ptypei<1000)) tmpy+=betai*Gravitational_CONST*(tempi-temp0);
	if((k_fs_solve)&(k_surf_model==2)){
		if((nmagi>0.1/hi)&(tmp_fsn>0)) curvi=tmp_fsn/tmp_fsd;
		else curvi=0;

		tmpx+=sigmai*curvi*nxi/rhoi;
		tmpy+=sigmai*curvi*nyi/rhoi;
	}

	Real A = 0.005;
	Real omega = 6.85;

	P3[i].ftotalx=tmpx-eulerx;
	P3[i].ftotaly=tmpy-eulery;
	
	P3[i].ftotal=sqrt(tmpx*tmpx+tmpy*tmpy);

	if(k_con_solve){
		if (enthalpy_eqn) P3[i].denthalpy=tmp_Rc*(ptypei!=-1);
		else{
			P3[i].dtemp=tmp_Rc/cpi*(ptypei!=-1)-eulert;
		}
	}
	if(k_concn_solve)	P3[i].dconcn=tmp_Rd;

}

////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_pressure_force2D(int_t inout,int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part3) return;
	if(!k_open_boundary && P1[i].i_type!=inout) return;
	if(k_open_boundary>0 && P1[i].buffer_type>0) return;
	if(P1[i].p_type>=1000)	return;	
	if(P1[i].p_type<=0)	return;	

	int_t ptypei;
	int_t icell,jcell;
	Real xi,yi,uxi,uyi,kci,eta;
	Real pi,hi,mi,rhoi,tempi,visi,betai;
	Real search_range,tmp_A,tmp_Rc,tmp_Rd;
	Real tmpx,tmpy,tmpn,tmpd;
	Real cpi, temp0;

	ptypei=P1[i].p_type;

	xi=P1[i].x;
	yi=P1[i].y;
	uxi=P1[i].ux;
	uyi=P1[i].uy;
	hi=P1[i].h;
	tempi=P1[i].temp;
	pi=P1[i].pres;
	mi=P1[i].m;
	rhoi=P1[i].rho;
	temp0=P1[i].temp0;

	tmp_A=calc_tmpA(hi);
	search_range=k_search_kappa*hi;	// search range


	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;

	tmpx=tmpy=0.0;

	for(int_t y=-P1[i].ncell;y<=P1[i].ncell;y++){
		for(int_t x=-P1[i].ncell;x<=P1[i].ncell;x++){
			int_t k=idx_cell(icell+x,jcell+y,0);

			if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))) continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					Real xj,yj,tdist; //rr
					int itype;
					xj=P1[j].x;
					yj=P1[j].y;
					itype=P1[j].i_type;

					if(P1[j].p_type<1000){
						if(itype!=4){
					tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj))+1e-20;
					Real tmp_hij;
					if(k_multi_type==3||k_multi_type==4)	tmp_hij=(P1[i].h+P1[j].h)/2;
					else	tmp_hij=P1[i].h;
					Real tmp_Aij=calc_tmpA(tmp_hij);

					search_range=k_search_kappa*fmax(P1[i].h,P1[j].h);
					if(tdist<search_range){
						int_t ptypej;
						Real tdwx,tdwy,uxj,uyj,mj,tempj,rhoj,pj,hj,kcj,sum_con_H, diffj, concnj,tmprd;
						Real nx_cj,ny_cj,nmag_cj,Phi_s,tmpnt;	// for surface tension
						Real volj;
						Real tdwxc, tdwyc;

						Real twij=calc_kernel_wij(tmp_Aij,tmp_hij,tdist);
						Real tdwij=calc_kernel_dwij(tmp_Aij,tmp_hij,tdist);

						tdwx=tdwij*(xi-xj)/tdist;
						tdwy=tdwij*(yi-yj)/tdist;

						tdwxc=tdwx*P3[i].Cm[0][0]+tdwy*P3[i].Cm[0][1];
						tdwyc=tdwx*P3[i].Cm[1][0]+tdwy*P3[i].Cm[1][1];

						ptypej=P1[j].p_type;
						uxj=P1[j].ux;
						uyj=P1[j].uy;
						mj=P1[j].m;
						tempj=P1[j].temp;
						rhoj=P1[j].rho;
						volj=P1[j].vol;
						pj=P1[j].pres;
						// if(P1[j].p_type<=0)	pj += rhoi*P1[j].dpres;

						if(k_fp_solve){
							// Real C_p=-2.0*mj/(rhoi*rhoj)*(pi*rhoj+pj*rhoi)/(rhoi+rhoj);
							// Real C_p=-2.0*mj/(rhoi*rhoj)*((pi+Pb)*rhoj+(pj+Pb)*rhoi)/(rhoi+rhoj);
							// if(P3[i].lambda < 0.8){
							// 	Real C_p=-mj*(pi+pj)/(rhoi*rhoj);

							// 	tmpx+=C_p*tdwx;
							// 	tmpy+=C_p*tdwy;
							// }else{
							// 	Real C_p=-mj*(-pi+pj)/(rhoi*rhoj);

							// 	tmpx+=C_p*tdwxc;
							// 	tmpy+=C_p*tdwyc;
							// }
							Real C_p=-mj*(pi+pj)/(rhoi*rhoj);

							tmpx+=C_p*tdwx;
							tmpy+=C_p*tdwy;
						}
						if(k_interface_solve){
							int_t flag;
							Real mrj,C_i;
							flag=(ptypei!=BOUNDARY)&&(ptypei!=MOVING)&&(ptypej!=BOUNDARY)&&(ptypej!=MOVING)&&(ptypei!=ptypej);
							mrj=mj/rhoj;
							// Real C_p=-2.0*mj/(rhoi*rhoj)*(pi*rhoj+pj*rhoi)/(rhoi+rhoj);
							// C_i=(abs(pi)*mri*mri+abs(pj)*mrj*mrj)*(flag);
							C_i=2.0*0.01*mj/(rhoi*rhoj)*(abs(pi)*rhoj+abs(pj)*rhoi)/(rhoi+rhoj)*(flag);
							tmpx+=C_i*tdwx;
							tmpy+=C_i*tdwy;

						}
					}
				}
				}
			}
			}
		}
	}

	P3[i].fpx=tmpx;
	P3[i].fpy=tmpy;
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_PPE3D(Real tdt, int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>i_type_crt) return;
	// if(k_open_boundary>0 && P1[i].buffer_type>0) return;
	if(P1[i].p_type>=1000)	return;		// Immersed Boundary Method
	if(P1[i].p_type<=0)	return;		// Immersed Boundary Method
	
	int_t icell,jcell,kcell;
	Real xi,yi,zi,uxi,uyi,uzi,rhoi;
	Real xistar,yistar,zistar,drhostar;
	Real search_range,hi,tmp_A;
	Real tmpx,tmpy,flt;
	Real bi, bix, biy,biz;
	Real Aij;
	Real AijPij;

	hi=P1[i].h;
	tmp_A=calc_tmpA(hi);
	search_range=k_search_kappa*hi;	// search range

	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;
	xistar=P1[i].x_star;
	yistar=P1[i].y_star;
	zistar=P1[i].z_star;
	uxi=P1[i].ux;
	uyi=P1[i].uy;
	uzi=P1[i].uz;
	rhoi=P1[i].rho;

	drhostar=0.0;
	bi = Aij = AijPij = 0.0;
	bix = 0.0;
	biy = 0.0;
	biz = 0.0;
	flt = 0.0;

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
			// int_t k=(icell+x)+k_NI*(jcell+y);
			int_t k=idx_cell(icell+x,jcell+y,kcell+z);

			if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					Real xj,yj,zj,uxj,uyj,uzj,tdist;
					Real xjstar,yjstar,zjstar,tdiststar;
					Real rhoj,mj,pj,rho_ref_j;
					Real volj;
					int p_type_j;
					int itype;

					itype=P1[j].i_type;
					mj=P1[j].m;

					if(P1[j].p_type<1000){
						if(itype!=4){
					xj=P1[j].x;
					yj=P1[j].y;
					zj=P1[j].z;
					xjstar=P1[j].x_star;
					yjstar=P1[j].y_star;
					zjstar=P1[j].z_star;
					uxj=P1[j].ux;
					uyj=P1[j].uy;
					uzj=P1[j].uz;
					rhoj=P1[j].rho;
					rho_ref_j=P2[j].rho_ref;
					volj=P1[j].vol;
					pj=P1[j].pres;

					tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj))+1e-20;
					tdiststar=sqrt((xistar-xjstar)*(xistar-xjstar)+(yistar-yjstar)*(yistar-yjstar)+(zistar-zjstar)*(zistar-zjstar))+1e-20;

					if(tdist<search_range){
						Real twij=calc_kernel_wij(tmp_A,hi,tdist);
						Real twijstar=calc_kernel_wij(tmp_A,hi,tdiststar);
						Real tdwij=calc_kernel_dwij(tmp_A,hi,tdist);

						Real tdwx=tdwij*(xi-xj)/tdist;
						Real tdwy=tdwij*(yi-yj)/tdist;
						Real tdwz=tdwij*(zi-zj)/tdist;
						// apply_gradient_correction_3D(P3[i].Cm,twij,tdwx,tdwy,tdwz,&tdwx,&tdwy,&tdwz);


						bix += mj/rhoj*(uxi-uxj)*tdwx;
						biy += mj/rhoj*(uyi-uyj)*tdwy;
						biz += mj/rhoj*(uzi-uzj)*tdwz;

						drhostar += (mj/rho_ref_j)*twijstar;
						flt += (mj/rhoj)*twij;


						Real c = -((xi-xj)*tdwx+(yi-yj)*tdwy+(zi-zj)*tdwz)/tdist/tdist;
						Aij += 2.0*mj/rhoj*c;
						AijPij += 2.0*mj/rhoj*c*pj;
					}
				}
			}
				}
			}
			}
		}
	}
	bi = rhoi*(bix+biy+biz)/tdt;
	// bi = -rhoi*(1.0-drhostar/flt)/tdt/tdt;
	P1[i].PPE1 = rhoi*(bix+biy+biz)/tdt;
	P1[i].PPE2 = -rhoi*(1.0-drhostar/flt)/tdt/tdt;


	// if(P3[i].lbl_surf==0)	P1[i].pres = (bi+AijPij)/Aij;
	// if(P3[i].lbl_surf!=0)	P1[i].pres = 0;
	P1[i].pres = (bi+AijPij)/Aij;
	// if(P1[i].pres<0.0){
	// 	P1[i].pres=0.0;
	// }
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_advection_force3D(int_t inout,int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>i_type_crt) return;
	// if(k_open_boundary>0 && P1[i].buffer_type>0) return;
	if(P1[i].p_type>=1000)	return;		// Immersed Boundary Method
	if(P1[i].p_type<=0)	return;		// Immersed Boundary Method

	int_t ptypei;
	int_t icell,jcell,kcell;
	Real xi,yi,zi,uxi,uyi,uzi,kci,eta;
	Real pi,hi,mi,mi8,mri,rhoi,tempi,visi,betai;
	Real diffi,concni;
	Real nxi,nyi,nzi,nmagi,sigmai;			// for surface tension
	Real nx_ci,ny_ci,nz_ci,nmag_ci,curvi; 	// for surface tension
	Real search_range,tmp_A,tmp_Rc,tmp_Rd;
	Real tmpx,tmpy,tmpz,tmpn,tmpd;
	Real tmp_fsn, tmp_fsd;
	Real tmpsx, tmpsy, tmpsz;
	Real eulerx, eulery, eulerz, eulert;
	Real cpi, temp0;

	ptypei=P1[i].p_type;

	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;
	uxi=P1[i].ux;
	uyi=P1[i].uy;
	uzi=P1[i].uz;
	hi=P1[i].h;
	tempi=P1[i].temp;
	pi=P1[i].pres;
	mi=P1[i].m;
	rhoi=P1[i].rho;

	tmp_A=calc_tmpA(hi);
	search_range=k_search_kappa*hi;	// search range

	if((k_fs_solve)&&(k_surf_model==2)){

		nxi=P3[i].nx;
		nyi=P3[i].ny;
		nzi=P3[i].nz;

		nmagi=P3[i].nmag;
		nx_ci=P3[i].nx_c;
		ny_ci=P3[i].ny_c;
		nz_ci=P3[i].nz_c;

		nmag_ci=P3[i].nmag_c;

		sigmai=sigma(tempi,ptypei);
	}

	if(k_con_solve){
		kci=conductivity(tempi,ptypei);
		cpi=specific_heat(tempi,ptypei);
		eta=0.001*hi;
	}
	if(k_concn_solve){
		concni=P1[i].concn;
		diffi=diffusion_coefficient(tempi,ptypei);
	}

	mi8=0.08/mi; // .. interface force
	mri=(mi/rhoi);

	// visi=viscosity(tempi,ptypei)+P3[i].vis_t;
	betai=thermal_expansion(tempi,ptypei);

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/k_dcell),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/k_dcell),k_NJ-1);}
	if((k_z_max==k_z_min)){kcell=0;}
	else{kcell=min(floor((zi-k_z_min)/k_dcell),k_NK-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

	tmpx=tmpy=tmpz=0.0;
	tmpn=0.0;
	tmpd=1.0;
	tmp_Rc=0.0;
	tmp_Rd=0.0;
	tmp_fsn=0.0;
	tmp_fsd=0.0;
	tmpsx=tmpsy=tmpsz=0.0;
	eulerx=eulery=eulerz=eulert=0.0;

	for(int_t z=-P1[i].ncell;z<=P1[i].ncell;z++){
		for(int_t y=-P1[i].ncell;y<=P1[i].ncell;y++){
			for(int_t x=-P1[i].ncell;x<=P1[i].ncell;x++){
				// int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
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
							int_t ptypej;
							Real tdwx,tdwy,tdwz,uxj,uyj,uzj,mj,tempj,rhoj,pj,hj,kcj,sum_con_H,diffj,concnj,tmprd;
							Real nx_cj,ny_cj,nz_cj,nmag_cj,Phi_s,tmpnt;	// for surface tension

							Real twij=calc_kernel_wij(tmp_A,hi,tdist);
							Real tdwij=calc_kernel_dwij(tmp_A,hi,tdist);
							Real wij_ipf=calc_kernel_wij_ipf(hi,tdist);
							Real wij_half=calc_kernel_wij_half(hi,tdist);
							
							tdwx=tdwij*(xi-xj)/tdist;
							tdwy=tdwij*(yi-yj)/tdist;
							tdwz=tdwij*(zi-zj)/tdist;


							if(k_kgc_solve>0){
								apply_gradient_correction_3D(P3[i].Cm,twij,tdwx,tdwy,tdwz,&tdwx,&tdwy,&tdwz);
							}

							ptypej=P1[j].p_type;
							uxj=P1[j].ux;
							uyj=P1[j].uy;
							uzj=P1[j].uz;
							mj=P1[j].m;
							tempj=P1[j].temp;
							rhoj=P1[j].rho;
							pj=P1[j].pres;
							hj=P1[j].h;

							if(k_fv_solve==Morris){
								Real visj,C_v;
								// visj=viscosity(tempj,ptypej)+P3[j].vis_t;
								//C_v=4*(mj/(rhoi*rhoj))*((visi*visj)/(visi+visj))*((xi-xj)*tdwx+(yi-yj)*tdwy+(zi-zj)*tdwz)/tdist/tdist;
								C_v=(xi-xj)*tdwx+(yi-yj)*tdwy+(zi-zj)*tdwz;
								C_v*=(visi*visj)/(visi+visj);
								C_v*=4*(mj/(rhoi*rhoj));
								C_v/=tdist;
								C_v/=tdist;

								tmpx+=C_v*(uxi-uxj);
								tmpy+=C_v*(uyi-uyj);
								tmpz+=C_v*(uzi-uzj);
								
							}else if(k_fv_solve==Monaghan){

								Real uij_xij=(uxi-uxj)*(xi-xj)+(uyi-uyj)*(yi-yj)+(uzi-uzj)*(zi-zj);
								Real h_ij,phi_ij,P_ij,P2_ij;
								// Real visj=viscosity(tempj,ptypej)+P3[j].vis_t;
								Real visj;

								// Monaghan type
								phi_ij=uij_xij;
								phi_ij/=(tdist*tdist+0.01*hi*hi);
								// P_ij=-2.0*(k_dim+2.0)*phi_ij*mj/rhoj;
								P_ij=-2.0*(k_dim+2.0)*phi_ij*mj/rhoj;
								Real nui,C_v;
								C_v=2.0*(visi*visj)/(visi+visj+1e-20);
								nui = C_v/rhoi;
								P_ij*= nui;

								if(P1[i].p_type<=0)	P_ij=0.0;
								if(P1[j].p_type<=0)	P_ij=0.0;

								tmpx+=-(P_ij)*tdwx;
								tmpy+=-(P_ij)*tdwy;
								tmpz+=-(P_ij)*tdwz;

							}
							if(k_fva_solve){
								Real uij_xij=(uxi-uxj)*(xi-xj)+(uyi-uyj)*(yi-yj)+(uzi-uzj)*(zi-zj);
								if(uij_xij<0){
									Real h_ij,phi_ij,P_ij;
									//
									h_ij=(hi+hj)*0.5;
									phi_ij=h_ij*uij_xij;
									phi_ij/=(tdist*tdist+0.01*h_ij*h_ij);
									P_ij=mi*phi_ij*(-Alpha*k_soundspeed+Beta*phi_ij);
									P_ij/=(rhoi+rhoj);
									P_ij*=0.5;
									//P_ij=mi*(-Alpha*k_soundspeed*phi_ij+Beta*phi_ij*phi_ij)/(rhoi+rhoj)*0.5;
									tmpx+=-(P_ij)*tdwx;
									tmpy+=-(P_ij)*tdwy;
									tmpz+=-(P_ij)*tdwz;
								}
							}

							if(k_interface_solve){
								int_t flag;
								Real mrj,C_i;
								//
								flag=(ptypei!=BOUNDARY)&&(ptypei!=MOVING)&&(ptypej!=BOUNDARY)&&(ptypej!=MOVING)&&(ptypei!=ptypej);
								mrj=mj/rhoj;
								C_i=abs(pi)*mri*mri+abs(pj)*mrj*mrj*(flag);
								C_i*=mi8*tdwij/tdist;

								//C_i=0.08/mi*(abs(pi)*(mi/rhoi)*(mi/rhoi)+abs(pj)*(mj/rhoj)*(mj/rhoj)*((ptypei!= BOUNDARY)&&(ptypei!=MOVING)&&(ptypej!=BOUNDARY)&&(ptypej!=MOVING)&&(ptypei!=ptypej)))*tdwij/tdist;
								// apply interface sharpness force just for the fluid particles (2017.06.22 jyb)
								tmpx+=C_i*(xj-xi);
								tmpy+=C_i*(yj-yi);
								tmpz+=C_i*(zj-zi);
							}
							if(k_fb_solve){
								if((ptypei==FLUID)&(ptypej!=FLUID)){
									Real fb_ij=k_c_repulsive/(tdist+1e-10)/(tdist+1e-10)*twij*(2*mj/(mi+mj));

									tmpx+=fb_ij*(xi-xj);
									tmpy+=fb_ij*(yi-yj);
									tmpz+=fb_ij*(zi-zj);
								}
							}
							if(k_fs_solve){
								if(k_surf_model==2){

									nx_cj=P3[j].nx_c;
									ny_cj=P3[j].ny_c;
									nz_cj=P3[j].nz_c;
	
									nmag_cj=P3[j].nmag_c;
									Phi_s=-(ptypei!= ptypej)+(ptypei==ptypej);
	
									tmpnt=((nx_ci/nmag_ci)-Phi_s*(nx_cj/nmag_cj))*(xj-xi);
									tmpnt+=((ny_ci/nmag_ci)-Phi_s*(ny_cj/nmag_cj))*(yj-yi);
									tmpnt+=((nz_ci/nmag_ci)-Phi_s*(nz_cj/nmag_cj))*(zj-zi);
	
									tmpnt*=k_dim*(mj/rhoj)*tdwij/tdist;
									tmp_fsn+=tmpnt;
									tmp_fsd+=(mj/rhoj)*tdist*abs(tdwij);
									}else	if(k_surf_model==1){
										Real Cs;
									if ((ptypei == 1) & (ptypej == 1))
									{
										Cs=s_ff1 * (A_ff1 * wij_half/(mi*(tdist + 1.0e-10)));
										Cs-=s_ff1 * (wij_ipf/(mi*(tdist + 1.0e-10)));
									}
									else if ((ptypei == 2) & (ptypej == 2))
									{
										Cs=s_ff2 * (A_ff2 * wij_half/(mi*(tdist + 1.0e-10)));
										Cs-=s_ff2 * (wij_ipf/(mi*(tdist + 1.0e-10)));
									}
									else if (((ptypei == 1) & (ptypej == 2)) || (ptypei == 2) & (ptypej == 1))
									{
										Cs=s_f1f2 * (A_f1f2 * wij_half/(mi*(tdist + 1.0e-10)));
										Cs-=s_f1f2 * (wij_ipf/(mi*(tdist + 1.0e-10)));
									}
									else if (((ptypei == 0) & (ptypej == 1)) || (ptypei == 1) & (ptypej == 0))
									{
										Cs=s_sf1 * (A_sf1 * wij_half/(mi*(tdist + 1.0e-10)));
										Cs-=s_sf1 * (wij_ipf/(mi*(tdist + 1.0e-10)));
									}
									else if (((ptypei == 0) & (ptypej == 2)) || (ptypei == 2) & (ptypej == 0))
									{
										Cs=s_sf2 * (A_sf2 * wij_half/(mi*(tdist + 1.0e-10)));
										Cs-=s_sf2 * (wij_ipf/(mi*(tdist + 1.0e-10)));
									}
									else if (((ptypei == -1) & (ptypej == 1)) || (ptypei == 1) & (ptypej == -1))
									{
										Cs=s_s2f1 * (A_s2f1 * wij_half/(mi*(tdist + 1.0e-10)));
										Cs-=s_s2f1 * (wij_ipf/(mi*(tdist + 1.0e-10)));
									}
									else if (((ptypei == -1) & (ptypej == 2)) || (ptypei == 2) & (ptypej == -1))
									{
										Cs=s_s2f2 * (A_s2f2 * wij_half/(mi*(tdist + 1.0e-10)));
										Cs-=s_s2f2 * (wij_ipf/(mi*(tdist + 1.0e-10)));
									}
									else
									{
										Cs=0.0;
									}
									tmpsx=Cs*(xi - xj);
									tmpsy=Cs*(yi - yj);
									tmpsz=Cs*(zi - zj);
									tmpx+=Cs*(xi - xj);
									tmpy+=Cs*(yi - yj);
									tmpz+=Cs*(zi - zj);
	
									}
							}
							if((P1[i].elix<1.0)||(P1[i].eliy<1.0)){
								eulerx += (uxi*(uxj-uxi)*tdwx+uyi*(uxj-uxi)*tdwy+uzi*(uxj-uxi)*tdwz)*mj/rhoj*(1-P1[i].elix);
								eulery += (uxi*(uyj-uyi)*tdwx+uyi*(uyj-uyi)*tdwy+uzi*(uyj-uyi)*tdwz)*mj/rhoj*(1-P1[i].eliy);
								eulerz += (uxi*(uzj-uzi)*tdwx+uyi*(uzj-uzi)*tdwy+uzi*(uzj-uzi)*tdwz)*mj/rhoj*(1-P1[i].eliz);

								if(k_con_solve){
									eulert += (uxi*(tempj-tempi)*tdwx+uyi*(tempj-tempi)*tdwy+uzi*(tempj-tempi)*tdwz)*mj/rhoj*(1-P1[i].elix);
								}
							}
							if(k_con_solve){
								kcj=conductivity(tempj, ptypej);
								sum_con_H=4.0*mj*kcj*kci*(tempi-tempj)*tdwij;
								sum_con_H/=(tdist+1e-10)*rhoi*rhoj*(kci+kcj);
								tmp_Rc+=sum_con_H;
							}
							if(k_concn_solve){
								concnj=P1[j].concn;
								diffj=diffusion_coefficient(tempj,ptypej);
								tmprd=mi*(diffi*rhoi+diffj*rhoj);
								tmprd*=((xi-xj)*tdwx+(yi-yj)*tdwy+(zi-zj)*tdwz);
								tmprd/=(rhoi*rhoj*(tdist*tdist+0.01*hi*hi));
								tmprd*=(concni-concnj)*(ptypei==ptypej);

								tmp_Rd+=tmprd;
							}
						}
					}
				}
			}
		}
	}
	// z-directional gravitational force
	if(k_fg_solve) tmpz+=-Gravitational_CONST;
	if((k_boussinesq_solve)&(ptypei>0)&(ptypei<1000)) tmpz+=betai*Gravitational_CONST*(tempi-temp0);
	if((k_fs_solve)&&(k_surf_model==2)){
		if((nmagi>0.1/hi)&(tmp_fsn>0)) curvi=tmp_fsn/tmp_fsd;
		else curvi=0;

		tmpx+=sigmai*curvi*nxi/rhoi;
		tmpy+=sigmai*curvi*nyi/rhoi;
		tmpz+=sigmai*curvi*nzi/rhoi;
	}

	P3[i].ftotalx=tmpx-eulerx;
	P3[i].ftotaly=tmpy-eulery;
	P3[i].ftotalz=tmpz-eulerz;

	if(k_con_solve){
		if (enthalpy_eqn) P3[i].denthalpy=tmp_Rc*(ptypei!=-1);
		else{
			P3[i].dtemp=tmp_Rc/cpi*(ptypei!=-1)-eulert;
		}
	}

	P3[i].fsx=tmpsx;
	P3[i].fsy=tmpsy;
	P3[i].fsz=tmpsz;

	P3[i].ftotal=sqrt(tmpx*tmpx+tmpy*tmpy+tmpz*tmpz);

	if(k_con_solve)	P3[i].denthalpy=tmp_Rc;
	if(k_concn_solve) P3[i].dconcn=tmp_Rd;
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_pressure_force3D(int_t inout,int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>i_type_crt) return;
	// if(k_open_boundary>0 && P1[i].buffer_type>0) return;
	if(P1[i].p_type>=1000)	return;		// Immersed Boundary Method
	if(P1[i].p_type<=0)	return;		// Immersed Boundary Method

	int_t ptypei;
	int_t icell,jcell,kcell;
	Real xi,yi,zi,uxi,uyi,uzi,kci,eta;
	Real pi,hi,mi,mi8,mri,rhoi,tempi,visi,betai;
	Real diffi,concni;
	Real nxi,nyi,nzi,nmagi,sigmai;			// for surface tension
	Real nx_ci,ny_ci,nz_ci,nmag_ci,curvi; 	// for surface tension
	Real search_range,tmp_A,tmp_Rc,tmp_Rd;
	Real tmpx,tmpy,tmpz,tmpn,tmpd;
	Real tmp_fsn, tmp_fsd;
	Real tmppx,tmppy,tmppz;

	ptypei=P1[i].p_type;

	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;
	uxi=P1[i].ux;
	uyi=P1[i].uy;
	uzi=P1[i].uz;
	hi=P1[i].h;
	tempi=P1[i].temp;
	pi=P1[i].pres;
	mi=P1[i].m;
	rhoi=P1[i].rho;

	tmp_A=calc_tmpA(hi);
	search_range=k_search_kappa*hi;	// search range

	if((k_fs_solve)&&(k_surf_model==2)){

		nxi=P3[i].nx;
		nyi=P3[i].ny;
		nzi=P3[i].nz;

		nmagi=P3[i].nmag;
		nx_ci=P3[i].nx_c;
		ny_ci=P3[i].ny_c;
		nz_ci=P3[i].nz_c;

		nmag_ci=P3[i].nmag_c;

		sigmai=sigma(tempi,ptypei);
	}

	if(k_con_solve){
		eta=0.001*hi;
		kci=conductivity(tempi,ptypei);
	}
	if(k_concn_solve){
		concni=P1[i].concn;
		diffi=diffusion_coefficient(tempi,ptypei);
	}

	mi8=0.08/mi; // .. interface force
	mri=(mi/rhoi);

	// visi=viscosity(tempi,ptypei)+P3[i].vis_t;
	betai=thermal_expansion(tempi,ptypei);

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/k_dcell),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/k_dcell),k_NJ-1);}
	if((k_z_max==k_z_min)){kcell=0;}
	else{kcell=min(floor((zi-k_z_min)/k_dcell),k_NK-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

	tmpx=tmpy=tmpz=0.0;
	tmpn=0.0;
	tmpd=1.0;
	tmp_Rc=0.0;
	tmp_Rd=0.0;
	tmp_fsn=0.0;
	tmp_fsd=0.0;
	tmppx=tmppy=tmppz=0.0;

	for(int_t z=-P1[i].ncell;z<=P1[i].ncell;z++){
		for(int_t y=-P1[i].ncell;y<=P1[i].ncell;y++){
			for(int_t x=-P1[i].ncell;x<=P1[i].ncell;x++){
				// int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
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
							int_t ptypej;
							Real tdwx,tdwy,tdwz,uxj,uyj,uzj,mj,tempj,rhoj,pj,hj,kcj,sum_con_H,diffj,concnj,tmprd;
							Real nx_cj,ny_cj,nz_cj,nmag_cj,Phi_s,tmpnt;	// for surface tension

							Real twij=calc_kernel_wij(tmp_A,hi,tdist);
							Real tdwij=calc_kernel_dwij(tmp_A,hi,tdist);

							tdwx=tdwij*(xi-xj)/tdist;
							tdwy=tdwij*(yi-yj)/tdist;
							tdwz=tdwij*(zi-zj)/tdist;


							if(k_kgc_solve>0){
								apply_gradient_correction_3D(P3[i].Cm,twij,tdwx,tdwy,tdwz,&tdwx,&tdwy,&tdwz);
							}

							ptypej=P1[j].p_type;
							uxj=P1[j].ux;
							uyj=P1[j].uy;
							uzj=P1[j].uz;
							mj=P1[j].m;
							tempj=P1[j].temp;
							rhoj=P1[j].rho;
							pj=P1[j].pres;
							hj=P1[j].h;


							if(k_fp_solve){
								Real C_p=-mj*(pi+pj)/(rhoi*rhoj);
								tmpx+=C_p*tdwx;
								tmpy+=C_p*tdwy;
								tmpz+=C_p*tdwz;
								tmppx=C_p*tdwx;
								tmppy=C_p*tdwy;
								tmppz=C_p*tdwz;

							}
						}
					}
				}
		}
	}
	}

	P3[i].fpx=tmpx;
	P3[i].fpy=tmpy;
	P3[i].fpz=tmpz;
	
}

// Apply advection force kernel (2D or 3D)
void advectionForce(
    dim3 b, dim3 t,
    int_t* g_str, int_t* g_end,
    part1* dev_SP1, part2* dev_SP2, part3* dev_P3
) {
    if (dim == 2)
        KERNEL_advection_force2D<<<b, t>>>(time, 1, g_str, g_end, dev_SP1, dev_SP2, dev_P3);
    if (dim == 3)
        KERNEL_advection_force3D<<<b, t>>>(1, g_str, g_end, dev_SP1, dev_SP2, dev_P3);
    cudaDeviceSynchronize();
}

// Apply pressure (PPE) force kernel (2D or 3D)
void pressureForce(
    dim3 b, dim3 t,
    int_t* g_str, int_t* g_end,
    part1* dev_SP1, part2* dev_SP2, part3* dev_P3
) {
	if (dim == 2)
        KERNEL_pressure_force2D<<<b, t>>>(1, g_str, g_end, dev_SP1, dev_SP2, dev_P3);
    if (dim == 3)
        KERNEL_pressure_force3D<<<b, t>>>(1, g_str, g_end, dev_SP1, dev_SP2, dev_P3);
    cudaDeviceSynchronize();
}

// Pressure Poisson Equation (PPE) Solver: applies the PPE CUDA kernel (2D or 3D)
void PPE(
    dim3 b, dim3 t,
    int_t* g_str, int_t* g_end,
    part1* dev_SP1, part2* dev_SP2, part3* dev_P3
) {
    if (dim == 2)
        KERNEL_PPE2D<<<b, t>>>(dt, g_str, g_end, dev_SP1, dev_SP2, dev_P3, 0);
    if (dim == 3)
        KERNEL_PPE3D<<<b, t>>>(dt, g_str, g_end, dev_SP1, dev_SP2, dev_P3);
    cudaDeviceSynchronize();
}
