////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_mass_update(int_t*g_str,int_t*g_end,part1*P1,part2*P2,int_t tcount)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>i_type_crt) return;
	if((P1[i].elix==1.0)&(P1[i].eliy==1.0)) return;
	if((P1[i].p_type<0)&(tcount>0)) return;
	if((k_solver_type==Icsph)&&(P1[i].p_type==1))	return;		// Immersed Boundary Method

	Real rhoi=P1[i].rho;
	P1[i].m=rhoi*P1[i].vol;
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_volume_update(int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3,int_t tcount)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>i_type_crt) return;
	if((P1[i].elix<1e-8)&(P1[i].eliy<1e-8)&(tcount>0)) return;
	if(P1[i].p_type>=1000)	return;		// Immersed Boundary Method
	// if((P1[i].p_type<0)&(tcount>0)) return;
	if((k_solver_type==Icsph)&&(P1[i].p_type==1))	return;		// Immersed Boundary Method
	
	int_t icell,jcell;
	Real xi,yi,uxi,uyi;
	Real rhoi,mi;
	// Real rho_ref_i;
	Real search_range,tmp_h,tmp_A,tmp_R;
	int p_type_i;
	Real tmpx,tmpy,tmprho, filt;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	p_type_i=P1[i].p_type;
	xi=P1[i].x;
	yi=P1[i].y;
	rhoi=P1[i].rho;
	mi=P1[i].m;
	// rho_ref_i=P2[i].rho_ref;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;

	tmp_R=filt=0.0;
	for(int_t y=-P1[i].ncell;y<=P1[i].ncell;y++){
		for(int_t x=-P1[i].ncell;x<=P1[i].ncell;x++){
			// int_t k=(icell+x)+k_NI*(jcell+y);
			int_t k=idx_cell(icell+x,jcell+y,0);

			if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))) continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					Real xj,yj,tdist;
					Real uxj, uyj, rhoj,mj;
					Real volj;
					int p_type_j;
					int itype;

					xj=P1[j].x;
					yj=P1[j].y;
					itype=P1[j].i_type;
					mj=P1[j].m;

					if(P1[j].p_type<1000){
						if(itype!=4){
					xj=P1[j].x;
					yj=P1[j].y;
					rhoj=P1[j].rho;
					volj=P1[j].vol;

					tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj))+1e-20;
					Real tmp_hij;
					if(k_multi_type==3||k_multi_type==4)	tmp_hij=(P1[i].h+P1[j].h)/2;
					else	tmp_hij=P1[i].h;
					Real tmp_Aij=calc_tmpA(tmp_hij);

					search_range=k_search_kappa*tmp_hij;

					if(tdist<search_range){
						Real twij=calc_kernel_wij(tmp_Aij,tmp_hij,tdist);
						
						tmp_R+=twij;

						filt+=mj/rhoj/P1[j].vol0*twij;
					}
				}
			}
				}
			}
		}
	}
	P1[i].vol=P1[i].vol0*1.0/tmp_R*filt;
	// P1[i].vol=1.0/tmp_R*filt;
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_continuity_norm2D(int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>i_type_crt) return;
	if(k_open_boundary>0 && P1[i].buffer_type>0) return;
	if(P1[i].p_type>=1000)	return;	
	// if(P1[i].p_type<=0)	return;	
	if((k_solver_type==Icsph)&&(P1[i].p_type==1))	return;
	
	int_t icell,jcell;
	Real xi,yi,uxi,uyi;
	Real rhoi;
	Real grad_rhoxi,grad_rhoyi;
	Real search_range,tmp_h,tmp_A;
	Real tmpx,tmpy,tmppx,tmppy,tmp_R;
	int p_type_i;

	p_type_i=P1[i].p_type;
	xi=P1[i].x;
	yi=P1[i].y;
	uxi=P1[i].ux;
	uyi=P1[i].uy;
	rhoi=P1[i].rho;
	Real rhoi_ref=P2[i].rho_ref;
	tmp_h=P1[i].h;
	grad_rhoxi=P1[i].grad_rhox;
	grad_rhoyi=P1[i].grad_rhoy;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;

	tmpx=tmpy=tmppx=tmppy=tmp_R=0.0;
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

						if(P1[j].p_type<1000){
							if(itype!=4){
						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj))+1e-20;

						Real tmp_hij;
						if(k_multi_type==3||k_multi_type==4)	tmp_hij=(P1[i].h+P1[j].h)/2;
						else	tmp_hij=P1[i].h;
						Real tmp_Aij=calc_tmpA(tmp_hij*1.0);

						search_range=k_search_kappa*fmax(P1[i].h,P1[j].h);

					if(tdist<search_range*1.0){
						Real mj,tdwx,tdwy,uxj,uyj,rhoj,grad_rhoxj,grad_rhoyj,phi_ij,tmprho,tmpr;
						Real tdwij=calc_kernel_dwij(tmp_Aij,tmp_hij*1.0,tdist);
						Real volj;

						p_type_j=P1[j].p_type;

						mj=P1[j].m;
						rhoj=P1[j].rho;
						Real rhoj_ref=P2[j].rho_ref;
						volj=P1[j].vol;
						uxj=P1[j].ux;
						uyj=P1[j].uy;
						grad_rhoxj=P1[j].grad_rhox;
						grad_rhoyj=P1[j].grad_rhoy;
						tmpr=0.0;

						tdwx=tdwij*(xi-xj)/tdist;
						tdwy=tdwij*(yi-yj)/tdist;

						// if(k_kgc_solve>0){
							Real twij=calc_kernel_wij(tmp_Aij,tmp_hij*1.0,tdist);
							// apply_gradient_correction_2D(P3[i].Cm,twij,tdwx,tdwy,&tdwx,&tdwy);
						// }

						// tmprho=mj*(rhoi/rhoj);
						tmprho=rhoi*mj/rhoj;
						tmpx+=(uxi-uxj)*tmprho*tdwx;
						tmpy+=(uyi-uyj)*tmprho*tdwy;
						tmppx+=uxi*mj/rhoj*rhoi_ref*(rhoi/rhoi_ref-rhoj/rhoj_ref)*tdwx*(1.0-P1[i].elix)*(p_type_i==p_type_j);
						tmppx+=-P1[i].dux*mj/rhoj*rhoi_ref*(rhoi/rhoi_ref-rhoj/rhoj_ref)*tdwx*(p_type_i==p_type_j);
						
						tmppy+=uyi*mj/rhoj*rhoi_ref*(rhoi/rhoi_ref-rhoj/rhoj_ref)*tdwy*(1.0-P1[i].eliy)*(p_type_i==p_type_j);
						tmppy+=-P1[i].duy*mj/rhoj*rhoi_ref*(rhoi/rhoi_ref-rhoj/rhoj_ref)*tdwy*(p_type_i==p_type_j);

						if(k_delSPH_solve>0) {
							// phi_ij=(grad_rhoxi+grad_rhoxj)*(xj-xi);
							// phi_ij+=(grad_rhoyi+grad_rhoyj)*(yj-yi);
							// phi_ij=(-0.5*phi_ij)*(k_delSPH_solve==Antuono)+(rhoj-rhoi);
							// tmpr=-2*delta*tmp_h*k_soundspeed*(mj/rhoj);
							// tmpr*=phi_ij*tdwij/tdist;
							// tmp_R+=tmpr*(P1[i].p_type==P1[j].p_type);

							// Density based Autuono model
							if(P1[j].p_type==2){
								Real rho0=fmax(1e-6,k_rho0_eos);
								double B = rho0*k_soundspeed*k_soundspeed/k_gamma;
								double K = P1[j].pres/B + 1.0;
								rhoj = P2[j].rho_ref*pow(K, 1.0/k_gamma);
							}

							phi_ij=(grad_rhoxi+grad_rhoxj)*(xj-xi);
							phi_ij+=(grad_rhoyi+grad_rhoyj)*(yj-yi);
							phi_ij=(-0.5*phi_ij)*(k_delSPH_solve==Antuono)+(rhoj/rhoj_ref-rhoi/rhoi_ref);
							tmpr=-2*delta*tmp_h*rhoi_ref*k_soundspeed*(volj);
							tmpr*=phi_ij*tdwij/tdist;
							tmp_R+=tmpr;

							// // Pressure based Antuono model
							// Real cc = k_soundspeed;
							// Real rho0=k_rho0_eos;
							// // if(P1[i].p_type==2)	cc = k_soundspeed*5.0;

							// Real rhoi_ref=P2[i].rho_ref;
							// Real presj = P1[j].pres;
							// Real presi = P1[i].pres;
							// if(P1[j].p_type<=0)	presj += rhoi*P1[j].dpres;

							// phi_ij=(grad_rhoxi+grad_rhoxj)*(xj-xi);
							// phi_ij+=(grad_rhoyi+grad_rhoyj)*(yj-yi);
							// phi_ij=(-0.5*phi_ij)*(k_delSPH_solve==Antuono)+k_gamma/cc/cc/rho0*(presj-presi);
							// tmpr=-2*delta*tmp_h*cc*rhoi_ref*mj/rhoj;
							// tmpr*=phi_ij*tdwij/tdist;
							// tmp_R+=tmpr;
						}
					}
				}
				}
			}
		}
	}
	}
	// P3[i].drho=(tmpx+tmpy)+(tmppx+tmppy)*(1-P1[i].eli)+tmp_R*(k_delSPH_solve>0);
	P3[i].drho=(tmpx+tmpy)+(tmppx+tmppy)+tmp_R*(k_delSPH_solve>0);
	// P3[i].drho=(tmpx+tmpy)+tmp_R*(k_delSPH_solve>0);

}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_continuity_norm3D(int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>i_type_crt) return;
	if(k_open_boundary>0 && P1[i].buffer_type>0) return;

	int_t icell,jcell,kcell;
	Real xi,yi,zi;
	Real rhoi;
	Real grad_rhoxi,grad_rhoyi,grad_rhozi;
	Real uxi,uyi,uzi;
	Real search_range,tmp_h,tmp_A;
	Real tmpx,tmpy,tmpz,tmp_R;
	int p_type_i;

	p_type_i=P1[i].p_type;

	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;
	uxi=P1[i].ux*(p_type_i>0);
	uyi=P1[i].uy*(p_type_i>0);
	uzi=P1[i].uz*(p_type_i>0);
	rhoi=P1[i].rho;

	grad_rhoxi=P1[i].grad_rhox;
	grad_rhoyi=P1[i].grad_rhoy;
	grad_rhozi=P1[i].grad_rhoz;
	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	if((k_z_max==k_z_min)){kcell=0;}
	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

	tmpx=tmpy=tmpz=tmp_R=0.0;
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
						int p_type_j;
						xj=P1[j].x;
						yj=P1[j].y;
						zj=P1[j].z;
						p_type_j=P1[j].p_type;

						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj))+1e-20;
						if(tdist<search_range){
							Real mj,tdwx,tdwy,tdwz,rhoj,uxj,uyj,uzj,tmprho,grad_rhoxj,grad_rhoyj,grad_rhozj,phi_ij,tmpr;

							Real tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);

							grad_rhoxj=P1[j].grad_rhox;
							grad_rhoyj=P1[j].grad_rhoy;
							grad_rhozj=P1[j].grad_rhoz;

							tdwx=tdwij*(xi-xj)/tdist;
							tdwy=tdwij*(yi-yj)/tdist;
							tdwz=tdwij*(zi-zj)/tdist;

							if(k_kgc_solve>0){
								Real twij=calc_kernel_wij(tmp_A,tmp_h,tdist);
								apply_gradient_correction_3D(P3[i].Cm,twij,tdwx,tdwy,tdwz,&tdwx,&tdwy,&tdwz);
							}

						  mj=P1[j].m;
							uxj=P1[j].ux*(p_type_j>0);
							uyj=P1[j].uy*(p_type_j>0);
							uzj=P1[j].uz*(p_type_j>0);
							rhoj=P1[j].rho;
							tmprho=mj*(rhoi/rhoj);
							tmpx+=(uxi-uxj)*tmprho*tdwx;
							tmpy+=(uyi-uyj)*tmprho*tdwy;
							tmpz+=(uzi-uzj)*tmprho*tdwz;

							if(k_delSPH_solve>0) {
								phi_ij=(grad_rhoxi+grad_rhoxj)*(xj-xi);
								phi_ij+=(grad_rhoyi+grad_rhoyj)*(yj-yi);
								phi_ij+=(grad_rhozi+grad_rhozj)*(zj-zi);
								phi_ij=(-0.5*phi_ij)*(k_delSPH_solve==Antuono)+(rhoj-rhoi);
								tmpr=-2*delta*tmp_h*k_soundspeed*(mj/rhoj);
								tmpr*=phi_ij*tdwij/tdist;
								tmp_R+=tmpr;
							}

						}
					}
				}
			}
		}
	}
	P3[i].drho=(tmpx+tmpy+tmpz)*(rhoi/P2[i].rho_ref)+tmp_R*(k_delSPH_solve>0);
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_mass_sum_norm2D(int_t*g_str,int_t*g_end,part1*P1,part2*P2)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>i_type_crt) return;
	if(k_open_boundary>0 && P1[i].buffer_type>0) return;
	if((k_solver_type==Icsph)&&(P1[i].p_type==1))	return;
	if(P1[i].p_type<=0)	return;	

	int_t icell,jcell;
	Real xi,yi;
	Real rho_ref_i;
	Real search_range,tmp_h,tmp_A,tmp_R;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	xi=P1[i].x;
	yi=P1[i].y;
	rho_ref_i=P2[i].rho_ref;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;

	tmp_R=0.0;
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

					tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj));
					if(tdist<search_range){
						Real twij,mj,rho_ref_j;
						twij=calc_kernel_wij(tmp_A,tmp_h,tdist);
						mj=P1[j].m;
						rho_ref_j=P2[j].rho_ref;
						tmp_R+=(mj/rho_ref_j)*twij;
					}
				}
			}
		}
	}
	P1[i].rho=rho_ref_i*tmp_R/P1[i].flt_s;
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_mass_sum_norm3D(int_t*g_str,int_t*g_end,part1*P1,part2*P2)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>i_type_crt) return;

	int_t icell,jcell,kcell;
	Real xi,yi,zi;
	Real rho_ref_i;
	Real search_range,tmp_h,tmp_A,tmp_R;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;
	rho_ref_i=P2[i].rho_ref;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	if((k_z_max==k_z_min)){kcell=0;}
	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

	tmp_R=0.0;
	for(int_t z=-P1[i].ncell;z<=P1[i].ncell;z++){
		for(int_t y=-P1[i].ncell;y<=P1[i].ncell;y++){
			for(int_t x=-P1[i].ncell;x<=P1[i].ncell;x++){
				// int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				int_t k=idx_cell(icell+x,jcell+y,kcell+z);

				if(k<0||k>=k_num_cells-1) continue;
				if(g_str[k]!=cu_memset){
					int_t fend=g_end[k];
					for(int_t j=g_str[k];j<fend;j++){
						Real xj,yj,zj,tdist;
						xj=P1[j].x;
						yj=P1[j].y;
						zj=P1[j].z;

						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));
						if(tdist<search_range){
							Real twij,mj,rho_ref_j;
							twij=calc_kernel_wij(tmp_A,tmp_h,tdist);
							mj=P1[j].m;
							rho_ref_j=P2[j].rho_ref;
							tmp_R+=(mj/rho_ref_j)*twij;
						}
					}
				}
			}
		}
	}
	P1[i].rho=rho_ref_i*tmp_R/P1[i].flt_s;
}