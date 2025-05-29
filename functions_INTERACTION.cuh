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
__global__ void KERNEL_interaction2D(int_t inout,int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part3) return;
	if(!k_open_boundary && P1[i].i_type!=inout) return;
	if(k_open_boundary>0 && P1[i].buffer_type>0) return;
	if(P1[i].p_type>=1000)	return;	
	if(P1[i].p_type<=0)	return;	
	if((k_solver_type==Icsph)&&(P1[i].p_type==1))	return;		// Immersed Boundary Method

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
	Real colorx, colory, gi_inv;
	Real ccx, ccy;
	Real eulerx, eulery, eulert;
	Real cpi, temp0;
	Real tmpfbdx,tmpfbdy;
	Real Di = P1[i].D;

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
	mi8=0.03/mi;
	mri=(mi/rhoi);

	visi=viscosity(tempi,Di,ptypei)+P3[i].vis_t;
	betai=thermal_expansion(tempi,ptypei);

	if((k_fs_solve)&(k_surf_model==2)){

		nxi=P3[i].nx;
		nyi=P3[i].ny;
		nmagi=P3[i].nmag;
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
	tmpfbdx=tmpfbdy=0.0;
	colorx=colory=0.0;
	ccx=ccy=0.0;
	curvi=gi_inv=0.0;

	for(int_t y=-P1[i].ncell;y<=P1[i].ncell;y++){
		for(int_t x=-P1[i].ncell;x<=P1[i].ncell;x++){
			// int_t k=(icell+x)+k_NI*(jcell+y);
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
						Real volj;
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
						volj=P1[j].vol;
						pj=P1[j].pres;
						hj=P1[j].h;
						Real Dj=P1[j].D;

						if(k_fp_solve){
							// Real C_p;
							Real C_p=-2.0*mj/(rhoi*rhoj)*((pi+Pb)*rhoj+(pj+Pb)*rhoi)/(rhoi+rhoj);

							// C_p=-mj*(pi+Pb+pj+Pb)/(rhoi*rhoj);
							// C_p=-mj*(pi+pj)/(rhoi*rhoj);
							
							// if(pi>=0)	C_p=-mj*(pi+pj)/(rhoi*rhoj);
							// if(pi<0)	C_p=-mj*(-pi+pj)/(rhoi*rhoj);
							tmpx+=C_p*tdwx;
							tmpy+=C_p*tdwy;
							// tmpx+=C_p*tdwxc;
							// tmpy+=C_p*tdwyc;
						}
						if(k_fv_solve>0){
							if(k_fv_solve==Morris){
								Real C_v;
								Real C_vv;
								Real visj=viscosity(tempj,Dj,ptypej)+P3[j].vis_t;
								C_v=(xi-xj)*tdwx+(yi-yj)*tdwy;
								C_v*=4*(mj/(rhoi*rhoj));
								C_v*=(visi*visj)/(visi+visj+1e-20);
								C_v/=tdist;
								C_v/=tdist;
								C_vv=(uxi-uxj)*tdwx+(uyi-uyj)*tdwy;
								C_vv*=4*(mj/(rhoi*rhoj));
								C_vv*=(visi*visj)/(visi+visj+1e-20);
								C_vv/=tdist;
								C_vv/=tdist;

								if(P1[i].p_type==0)	C_v=0.0;
								if(P1[j].p_type==0)	C_v=0.0;
								if(P1[i].p_type==0)	C_vv=0.0;
								if(P1[j].p_type==0)	C_vv=0.0;

								tmpx+=C_v*(uxi-uxj)+C_vv*(xi-xj);
								tmpy+=C_v*(uyi-uyj)+C_vv*(yi-yj);

								// Real C_v;
								// Real visj=viscosity(tempj,Dj,ptypej)+P3[j].vis_t;
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
						if(k_interface_solve){
							int_t flag;
							Real mrj,C_i;
							flag=(ptypei!=BOUNDARY)&&(ptypei!=MOVING)&&(ptypej!=BOUNDARY)&&(ptypej!=MOVING)&&(ptypei!=ptypej);
							Real C_p=-0.1*2.0*mj/(rhoi*rhoj)*(abs(pi+Pb)*rhoj+abs(pj+Pb)*rhoi)/(rhoi+rhoj)*flag;
							tmpx+=C_p*(xj-xi);
							tmpy+=C_p*(yj-yi);

						}
						if(k_fb_solve){
							if((ptypei==2)&(ptypej<=0)){
								Real nx_w = P3[j].nx_w;
								Real ny_w = P3[j].ny_w;
								Real gamma_p=1.0;
								Real ex = (xj-xi)/tdist;
								Real ey = (yj-yi)/tdist;
								Real delt=2.0/(hi/1.5)*(ex*nx_w+ey*ny_w)*tdist;
								Real lamb=(delt<1.0)*(1.0-delt)*(1.0-delt)*0.01;
								Real penalty=gamma_p*lamb*abs((pi+9.8*0.6*1000.0)*(ex*nx_w+ey*ny_w)*tdist);
								Real fb_ij=-2.0*mj/rhoj/rhoi*penalty;
								tmpx+=fb_ij*tdwij*nx_w;
								tmpy+=fb_ij*tdwij*ny_w;
								tmpfbdx+=fb_ij*tdwij*nx_w;
								tmpfbdy+=fb_ij*tdwij*ny_w;
							}
						}
						if(k_fs_solve){
							if(k_surf_model==2){

							// Real nxj,nyj,nmagj,Phi_s,tmpnt;	// for surface tension

							// nxj=P3[j].nx;
							// nyj=P3[j].ny;

							// nmagj=P3[j].nmag;
							// Phi_s=-(ptypei!= ptypej)+(ptypei==ptypej);

							// // tmpnt=(-(nxi/nmagi)+Phi_s*(nxj/nmagj))*tdwx;
							// // tmpnt+=(-(nyi/nmagi)+Phi_s*(nyj/nmagj))*tdwy;

							// // tmpnt*=k_dim*(mj/rhoj);
							// // tmp_fsn+=tmpnt;
							// // tmp_fsd+=(mj/rhoj)*tdist*abs(tdwij);

							// tmpnt=(-(nxi/nmagi)+Phi_s*(nxj/nmagj))*tdwxc;
							// tmpnt+=(-(nyi/nmagi)+Phi_s*(nyj/nmagj))*tdwyc;

							// tmpnt*=(mj/rhoj);
							// tmp_fsn+=tmpnt;
							// tmp_fsd=1.0;

							Real nxj,nyj,nzj,nmagj;

							nxj=P3[j].nx;
							nyj=P3[j].ny;
							nmagj=P3[j].nmag;

							int_t flag = ((nmagi>0)&(nmagj>0)&(nxi*nxj+nyi*nyj>=-1.0/2.0))*(P1[i].p_type==P1[j].p_type);

							curvi+=flag*((nxj-nxi)*tdwxc+(nyj-nyi)*tdwyc)*mj/rhoj;

							colorx+=tdwx*mj/rhoj*(P1[i].p_type==P1[j].p_type);
							colory+=tdwy*mj/rhoj*(P1[i].p_type==P1[j].p_type);
							ccx+=tdwx*mj/rhoj*(P1[i].p_type==P1[j].p_type);
							ccy+=tdwy*mj/rhoj*(P1[i].p_type==P1[j].p_type);

							gi_inv+=2.0*twij*mj/rhoj*(P1[i].p_type==P1[j].p_type);
							
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
	// y-directional gravitational force
	if(k_fg_solve) tmpy+=-Gravitational_CONST;
	if((k_boussinesq_solve)&(ptypei>0)&(ptypei<1000)) tmpy+=betai*Gravitational_CONST*(tempi-temp0);
	if((k_fs_solve)&(k_surf_model==2)){
		// if(nmagi>rhoi) curvi=tmp_fsn/tmp_fsd;
		// else curvi=0;
		// P3[i].curv=curvi;
		
		// tmpx+=-sigmai*curvi*nxi/rhoi;
		// tmpy+=-sigmai*curvi*nyi/rhoi;

		Real color = 0.0;
		Real cc = 0.0;
		Real gi = 0.0;

		gi = 1.0/(gi_inv+1e-10);
		color =2.0*fmax(gi,1.0)*sqrt(colorx*colorx + colory*colory );
		cc =sqrt(ccx*ccx + ccy*ccy);

		P3[i].color = color;
		P3[i].curv=curvi*(cc>250.0);

		// P3[i].fsx=-sigmai*curvi*nxi*color/rhoi*(cc>250.0)*(curvi>0);
		// P3[i].fsy=-sigmai*curvi*nyi*color/rhoi*(cc>250.0)*(curvi>0);

		tmpx+=-sigmai*curvi*nxi*color/rhoi*(cc>250.0)*(curvi>0);
		tmpy+=-sigmai*curvi*nyi*color/rhoi*(cc>250.0)*(curvi>0);
	}

	P3[i].ftotalx=tmpx-eulerx;
	P3[i].ftotaly=tmpy-eulery;
	P3[i].fbdx=tmpfbdx;
	P3[i].fbdy=tmpfbdy;
	
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
__global__ void KERNEL_interaction3D(int_t inout,int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part3) return;
	if(P1[i].i_type!=inout) return;

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

	if(k_fs_solve){

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
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	if((k_z_max==k_z_min)){kcell=0;}
	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

	tmpx=tmpy=tmpz=0.0;
	tmpn=0.0;
	tmpd=1.0;
	tmp_Rc=0.0;
	tmp_Rd=0.0;
	tmp_fsn=0.0;
	tmp_fsd=0.0;

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
							}
							if(k_fv_solve){
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
							}
							if(k_boussinesq_solve){
								if((ptypei!=BOUNDARY)&(ptypei!=MOVING)){
									tmpn+=mj*(tempj-tempi)*twij*(ptypei==ptypej)/rhoj;
									tmpd+=mj*twij*(ptypei==ptypej)/rhoj;
								}
							}
							// if(k_con_solve && !(k_EOS_type==Ideal)){
							// 		kcj=conductivity(tempj, ptypej);
							// 		sum_con_H=4.0*mj*kcj*kci*(tempi-tempj)*tdwij;
							// 		sum_con_H/=(tdist+eta*eta)*rhoi*rhoj*(kci+kcj);
							// 		tmp_Rc+=sum_con_H;
							// }
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
	if(k_boussinesq_solve) tmpz+=-betai*Gravitational_CONST*(tmpn/tmpd);
	if(k_fs_solve){
		if((nmagi>0.1/hi)&(tmp_fsn>0)) curvi=tmp_fsn/tmp_fsd;
		else curvi=0;

		tmpx+=sigmai*curvi*nxi/rhoi;
		tmpy+=sigmai*curvi*nyi/rhoi;
		tmpz+=sigmai*curvi*nzi/rhoi;
	}

	P3[i].ftotalx=tmpx;
	P3[i].ftotaly=tmpy;
	P3[i].ftotalz=tmpz;

	P3[i].ftotal=sqrt(tmpx*tmpx+tmpy*tmpy+tmpz*tmpz);

	if(k_con_solve)	P3[i].denthalpy=tmp_Rc;
	if(k_concn_solve) P3[i].dconcn=tmp_Rd;
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_vorticity_2D(int_t*g_str,int_t*g_end,part1*P1,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part3) return;
	if(P1[i].i_type==3) return;

	int_t ptypei;
	int_t icell,jcell,kcell;
	Real xi,yi,zi,uxi,uyi,uzi;
	Real tmp_A,hi,search_range,mi,rhoi;
	Real vortx,vorty,vortz;

	ptypei=P1[i].p_type;

	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;
	uxi=P1[i].ux;
	uyi=P1[i].uy;
	uzi=P1[i].uz;
	hi=P1[i].h;
	mi=P1[i].m;
	rhoi=P1[i].rho;


	tmp_A=calc_tmpA(hi);
	search_range=k_search_kappa*hi;	// search range



	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;

	vortx=vorty=vortz=0.0;

	for(int_t y=-P1[i].ncell;y<=P1[i].ncell;y++){
		for(int_t x=-P1[i].ncell;x<=P1[i].ncell;x++){
			// int_t k=(icell+x)+k_NI*(jcell+y);
			int_t k=idx_cell(icell+x,jcell+y,0);
			if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))) continue;
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
							Real tdwx,tdwy,tdwz,uxj,uyj,uzj,mj,rhoj;

							Real twij=calc_kernel_wij(tmp_A,hi,tdist);
							Real tdwij=calc_kernel_dwij(tmp_A,hi,tdist);

							tdwx=tdwij*(xi-xj)/tdist;
							tdwy=tdwij*(yi-yj)/tdist;
							tdwz=tdwij*(zi-zj)/tdist;


							if(k_kgc_solve>0){
								apply_gradient_correction_3D(P3[i].Cm,twij,tdwx,tdwy,tdwz,&tdwx,&tdwy,&tdwz);
							}

							mj=P1[j].m;
							rhoj=P1[j].rho;

							uxj=P1[j].ux;
							uyj=P1[j].uy;
							uzj=P1[j].uz;
							ptypej=P1[j].p_type;

							vortz+=-mj/rhoj*((uyi-uyj)*tdwx-(uxi-uxj)*tdwy)*(ptypei==ptypej);
						}
					}
				}
		}
	}
	P1[i].vortz=vortz;

}

void drag_coefficient(part1*P1,part3*P3,Real*c_drag,Real*c_lift,Real*Cd,Real*Cl){
	int_t nop = num_part3;
	int_t nop_r = 0;

	// for(int i=0;i<nop;i++) if(P1[i].p_type>=1000)	nop_r++;

	Real fd, fl;
	fd=fl=0.0;

	// Real fibm = 1000.0*
	for(int i=0;i<nop;i++){
		if((P1[i].p_type>=1000)&&(P1[i].i_type!=3)){
			fd+=(-P1[i].fbx*P1[i].vol);
			fl+=(-P1[i].fby*P1[i].vol);
		}
		// if((P1[i].p_type==0)){
		// 	fd+=(P1[i].fcx*P1[i].m);
		// 	fl+=(P1[i].fcy*P1[i].m);
		// }
	}

	Real cd,cl;

	cd = fd/(0.5*1000.0*0.1*0.1*0.02);
	cl = fl/(0.5*1000.0*0.1*0.1*0.02);

	*c_drag = cd;
	*c_lift = cl;

	char FileName[512];
	sprintf(FileName,"./record/coefficient.txt",count);
	FILE*outFile;
	outFile=fopen(FileName,"w");

	fprintf(outFile,"coefficient \n\n");

	for (int i=0; i<(time_end/time_output); i++){
	fprintf(outFile, "%d\ttime=%f\tcd=%f\tcl=%f\n",i, i*time_output, Cd[i], Cl[i]);
	}
	fclose(outFile);
}

// void pressureprobe(Real ttime,part1*P1,part3*P3,Real*time0,Real*p0,Real*p1,Real*p2,Real*p3,Real*ttime0,Real*PP0,Real*PP1,Real*PP2,Real*PP3){
// 	int_t nop = num_part2;
// 	int_t nop_r = 0;

// 	Real pres0, num0;
// 	Real pres1, num1;
// 	Real pres2, num2;
// 	Real pres3, num3;
// 	pres0=pres1=pres2=pres3=0.0;
// 	num0=num1=num2=num3=0.0;

// 	Real height = 0.6;
// 	Real LW = 3.366*height;
// 	Real nonp = 1000.0*9.8*height;
// 	Real h0 = P1[0].h;
// 	Real tA;
// 	tA=3.546881588905096/(kappa*h0)/(kappa*h0);

// 	for(int i=0;i<nop;i++){
// 		Real xi = P1[i].x;
// 		Real yi = P1[i].y;
// 		Real twij=0.0;
// 		Real tdist=0.0;
// 		Real tR=0.0;

// 		if((abs(yi-0.0)<kappa*h0)&&(abs(xi-LW)<kappa*h0)&&(P1[i].i_type<3)){
// 			tdist = sqrt(abs(yi-0.0)*abs(yi-0.0)+abs(xi-LW)*abs(xi-LW));
// 			tR=tdist/h0/kappa;
// 			twij=(tR<1)*tA*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1+8*tR+25*tR*tR+32*tR*tR*tR);
// 			pres0+=P1[i].pres*twij;
// 			num0+=twij;
// 		}

// 		// if((abs(yi-0.16)<kappa*h0)&&(abs(xi-LW)<kappa*h0)){
// 		if((abs(yi-0.1)<kappa*h0)&&(abs(xi-LW)<kappa*h0)&&(P1[i].i_type<3)){
// 			tdist = sqrt(abs(yi-0.1)*abs(yi-0.1)+abs(xi-LW)*abs(xi-LW));
// 			tR=tdist/h0/kappa;
// 			twij=(tR<1)*tA*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1+8*tR+25*tR*tR+32*tR*tR*tR);
// 			pres1+=P1[i].pres*twij;
// 			num1+=twij;
// 		}

// 		if((abs(yi-0.584)<kappa*h0)&&(abs(xi-LW)<kappa*h0)&&(P1[i].i_type<3)){
// 			tdist = sqrt(abs(yi-0.584)*abs(yi-0.584)+abs(xi-LW)*abs(xi-LW));
// 			tR=tdist/h0/kappa;
// 			twij=(tR<1)*tA*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1+8*tR+25*tR*tR+32*tR*tR*tR);
// 			pres2+=P1[i].pres*twij;
// 			num2+=twij;
// 		}

// 		if((abs(yi-1.0)<kappa*h0)&&(abs(xi-LW)<kappa*h0)&&(P1[i].i_type<3)){
// 			tdist = sqrt(abs(yi-1.0)*abs(yi-1.0)+abs(xi-LW)*abs(xi-LW));
// 			tR=tdist/h0/kappa;
// 			twij=(tR<1)*tA*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1+8*tR+25*tR*tR+32*tR*tR*tR);
// 			pres3+=P1[i].pres*twij;
// 			num3+=twij;
// 		}
// 	}

// 	*p0=pres0/(num0+1e-10)/nonp;
// 	*p1=pres1/(num1+1e-10)/nonp;
// 	*p2=pres2/(num2+1e-10)/nonp;
// 	*p3=pres3/(num3+1e-10)/nonp;
// 	*time0=ttime;

// 	char FileName[512];
// 	sprintf(FileName,"./record/probe.txt",count);
// 	FILE*outFile;
// 	outFile=fopen(FileName,"w");

// 	fprintf(outFile,"probe \n\n");

// 	for (int i=0; i<(time_end/time_output); i++){
// 	fprintf(outFile, "%d\tt=%f\tp0=%f\tp1=%f\tp2=%f\tp3=%f\n",i, ttime0[i]*sqrt(9.8/height),PP0[i],PP1[i],PP2[i],PP3[i]);
// 	}
// 	fclose(outFile);
// }

void pressureprobe(Real ttime,part1*P1,part3*P3,Real*time0,Real*p0,Real*p1,Real*p2,Real*p3,Real*ttime0,Real*PP0,Real*PP1,Real*PP2,Real*PP3){
	int_t nop = num_part2;
	int_t nop_r = 0;

	Real pres0, num0;
	Real pres1, num1;
	Real pres2, num2;
	Real pres3, num3;
	pres0=pres1=pres2=pres3=0.0;
	num0=num1=num2=num3=0.0;
	Real x_m2 = -0.3;
	Real y_m2 = 0.02;
	Real x_m1 = 0.3-0.02;
	Real y_m1 = 0.3;
	Real h_m1 = 0.0;

	Real height = 0.3;
	Real LW = 3.366*height;
	Real nonp = 1000.0*9.8*height;
	Real h0 = P1[0].h;
	Real tA;
	tA=3.546881588905096/(kappa*h0)/(kappa*h0);

	for(int i=0;i<nop;i++){
		Real xi = P1[i].x;
		Real yi = P1[i].y;
		Real twij=0.0;
		Real tdist=0.0;
		Real tR=0.0;

		if(((abs(yi-y_m2))<kappa*h0)&&(abs(xi-x_m2)<kappa*h0)&&(P1[i].i_type<3)){
			tdist = sqrt(abs(yi-y_m2)*abs(yi-y_m2)+abs(xi-x_m2)*abs(xi-x_m2));
			tR=tdist/h0/kappa;
			twij=(tR<1)*tA*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1+8*tR+25*tR*tR+32*tR*tR*tR);
			pres0+=P1[i].pres*twij;
			num0+=twij;
		}

		if((abs(xi-x_m1)<h0/1.5)&&(P1[i].i_type<3)&&(P1[i].p_type==1)){
			if(yi>h_m1) h_m1 = yi;
		}

	}

	*p0=pres0/(num0+1e-10)/nonp;
	*p1=h_m1;
	*time0=ttime;

	char FileName[512];
	sprintf(FileName,"./record/probe.txt",count);
	FILE*outFile;
	outFile=fopen(FileName,"w");

	fprintf(outFile,"probe \n\n");

	for (int i=0; i<(time_end/time_output); i++){
	fprintf(outFile, "%d\tt=%f\tp0=%f\th0=%f\n",i, ttime0[i]*sqrt(9.8/height),PP0[i],PP1[i]);
	}
	fclose(outFile);
}
void error_analysis(Real ttime,part1*P1,part3*P3,Real*time0,Real*L2_norm,Real*L1_norm,Real*Linf_norm,Real*ttime0,Real*L2norm,Real*L1norm,Real*Linfnorm){
	int_t nop = num_part2;
	int_t nop_r = 0;

	Real ll2, ll1, llinf;
	ll2=ll1=llinf=0.0;
	Real error_=0.0;

	for(int i=0;i<nop;i++){
		if((P1[i].p_type>0&&P1[i].buffer_type==0)){
			nop_r=nop_r+1;
			Real uyi=P1[i].uy;
			Real xi=P1[i].x;
			Real tempi=P1[i].temp;
			Real rho=1.0;
			Real mu=1.0;
			Real mu2=1.0;
			Real d=1.0;
			Real u0=1.0;
			Real f=2.0;
			Real N = 4.0;
			Real uavg = 2/d*pow((1/mu*f),(1/N))*N/(N+1)*(d/2*pow((d/2),(1/N+1))-N/(1+2*N)*pow((d/2),(1/N+2)));
			Real velth;
			velth= pow((1/mu*f),(1/N))*N/(N+1)*(pow((d/2),(1/N+1))-pow((abs(xi)),(1/N+1)));
			Real error = abs(uyi-velth)/uavg;
			ll2+=error*error;
			ll1+=error;
			llinf=fmax(error,error_);
			error_=error;
			P1[i].uanalytic=velth/uavg;
		}
	}

*L2_norm=sqrt(ll2)/nop_r;
*L1_norm=ll1/nop_r;
*Linf_norm=llinf;
*time0=ttime;

	char FileName[512];
	sprintf(FileName,"./record/error.txt",count);
	FILE*outFile;
	outFile=fopen(FileName,"w");

	fprintf(outFile,"error \n\n");

	for (int i=0; i<(time_end/time_output); i++){
	fprintf(outFile, "%d\ttime=%f\tl2=%f\tl1=%f\tlinf=%f\n",i, ttime0[i], L2norm[i], L1norm[i], Linfnorm[i]);
	}
	fclose(outFile);
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_normalvector_atwall2D(int_t*g_str,int_t*g_end,part1*P1,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part3) return;
	if(P1[i].p_type>=1000)	return;		// Immersed Boundary Method
	if(P1[i].p_type>0)	return;		// Immersed Boundary Method

	int_t icell,jcell,non,ptypei;
	Real xi,yi;
	Real rn;
	Real tmpx,tmpy;
	Real search_range;
	Real hi,tmp_A;
	hi=P1[i].h;

	tmp_A=calc_tmpA(hi);

	search_range=k_search_kappa*P1[i].h;

	xi=P1[i].x;
	yi=P1[i].y;
	ptypei=P1[i].p_type;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;

	tmpx=tmpy=0.0;
	non=0;

	for(int_t y=-P1[i].ncell;y<=P1[i].ncell;y++){
		for(int_t x=-P1[i].ncell;x<=P1[i].ncell;x++){
			int_t k=(icell+x)+k_NI*(jcell+y);
			if(k<0||k>=k_num_cells-1) continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					Real xj,yj,rr,tdist;
					int_t	itypej,ptypej;

					xj=P1[j].x;
					yj=P1[j].y;
					itypej=P1[j].i_type;
					ptypej=P1[j].p_type;

					rr=(xi-xj)*(xi-xj)+(yi-yj)*(yi-yj);
					tdist=sqrt(rr);
					if(itypej!=4){
						if(ptypej<1000){

					if(tdist<search_range){
						Real rd=1.0/(tdist+1e-10);
						Real tdwij=calc_kernel_dwij(tmp_A,hi,tdist);
						Real tdwx, tdwy;
						Real rhoj, mj;
						rhoj = P1[j].rho;
						mj = P1[j].m;

						tdwx=tdwij*(xi-xj)/(tdist+1e-10);
						tdwy=tdwij*(yi-yj)/(tdist+1e-10);

						int_t flag = (ptypej<=0);
						tmpx+=-tdwx*flag;
						tmpy+=-tdwy*flag;
					}
				}
			}
			}
			}
		}
	}
	P3[i].nx_w=tmpx/sqrt(tmpx*tmpx+tmpy*tmpy);
	P3[i].ny_w=tmpy/sqrt(tmpx*tmpx+tmpy*tmpy);
}