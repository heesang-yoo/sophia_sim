////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_prep2D(int_t*g_str,int_t*g_end,part1*P1, part2*P2, part3*P3, int_t tcount)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part3) return;
	if(P1[i].i_type>i_type_crt) return;
	if(P1[i].p_type>=1000)	return;		// Immersed Boundary Method

	int_t icell,jcell;
	int_t ptypei=P1[i].p_type;

	Real xi,yi;
	Real uxi,uyi;
	Real mi=P1[i].m;
	Real rhoi=P1[i].rho;
	Real cci=P3[i].cc;
	Real tmp_h,tmp_A,search_range;
	Real tmp_flt,tmp_SR;
	Real tmp_rhox,tmp_rhoy;
	Real tmp_ncx, tmp_ncy, tmp_nx, tmp_ny;
	Real tvis_t=0.0,th;
	Real tmp_uxx, tmp_uxy, tmp_uyx, tmp_uyy;
	Real li = P3[i].lambda;
	Real grad_lix, grad_liy;
	Real shearxx, shearxy, shearyx, shearyy;	//non-newtonian

	xi=P1[i].x;
    yi=P1[i].y;

	uxi=P1[i].ux;
	uyi=P1[i].uy;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	th=tmp_h*L_SPS;
	search_range=k_search_kappa*tmp_h;	// search range

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}

	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;

	// 변수 초기화
	tmp_flt=tmp_SR=tmp_rhox=tmp_rhoy=0.0;
	tmp_nx=tmp_ny=tmp_ncx=tmp_ncy=0.0;
	tmp_uxx=tmp_uxy=tmp_uyx=tmp_uyy=0.0;
	grad_lix=grad_liy=0.0;
	shearxx=shearxy=shearyx=shearyy=0.0;	//non-newtonian

	// 계산
	for(int_t y=-P1[i].ncell;y<=P1[i].ncell;y++){
		for(int_t x=-P1[i].ncell;x<=P1[i].ncell;x++){
			// int_t k=(icell+x)+k_NI*(jcell+y);
			int_t k=idx_cell(icell+x,jcell+y,0);

			if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))) continue;

			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					Real xj,yj,uxj,uyj,uij2,mj,rhoj,ccj,tdwx,tdwy,tmp_wij,tmp_dwij,tdist,tmp_val;
					int_t ptypej;
					int itype;
					Real volj;

					xj=P1[j].x;
					yj=P1[j].y;
					uxj=P1[j].ux;
					uyj=P1[j].uy;
					mj=P1[j].m;
					rhoj=P1[j].rho;
					ptypej=P1[j].p_type;
					ccj=P3[j].cc;
					itype=P1[j].i_type;
					volj=P1[j].vol;
					Real lj = P3[j].lambda;

					tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj))+1e-20;
					if(itype!=4){
					if(P1[j].p_type<1000){
						Real tmp_hij;
						if(k_multi_type==3||k_multi_type==4)	tmp_hij=(P1[i].h+P1[j].h)/2;
						else	tmp_hij=P1[i].h;
						Real tmp_Aij=calc_tmpA(tmp_hij*1.0);
	
						search_range=k_search_kappa*fmax(P1[i].h,P1[j].h);
					if(tdist<search_range*1.0){

						tmp_wij=calc_kernel_wij(tmp_Aij,tmp_hij*1.0,tdist);
						tmp_dwij=calc_kernel_dwij(tmp_Aij,tmp_hij*1.0,tdist);

						// dwij
						tdwx=tmp_dwij*(xi-xj)/tdist;
						tdwy=tmp_dwij*(yi-yj)/tdist;

						
						Real tdwxc=P3[i].Cm[0][0]*tdwx+P3[i].Cm[0][1]*tdwy;
						Real tdwyc=P3[i].Cm[1][0]*tdwx+P3[i].Cm[1][1]*tdwy;

						// filter
						if((tcount%k_freq_filt)==0) tmp_flt+=mj/rhoj*tmp_wij;

						// strain rate
						if((k_fv_solve==1)&&(k_turbulence_model!=Laminar))
						{
							uij2=(uxi-uxj)*(uxi-uxj)+(uyi-uyj)*(uyi-uyj);
							tmp_val=-0.5*mj*(rhoi+rhoj)*uij2;
							tmp_val/=(rhoi*rhoj*tdist*tdist);
							tmp_SR+=tmp_val*(xi-xj)*tdwx+tmp_val*(yi-yj)*tdwy;
						}

						// gradient rho (for delta-sph)
						if(k_delSPH_solve==Antuono)
						{
							// tmp_rhox+=-(rhoj-rhoi)*mj/rhoj*tdwx*(P1[i].p_type==P1[j].p_type);
							// tmp_rhoy+=-(rhoj-rhoi)*mj/rhoj*tdwy*(P1[i].p_type==P1[j].p_type);

							// Density based Autuono model
							Real rhoi_ref=P2[i].rho_ref;
							Real rhoj_ref=P2[j].rho_ref;

							if(P1[j].p_type==2){
								Real rho0=fmax(1e-6,k_rho0_eos);
								double B = rho0*k_soundspeed*k_soundspeed/k_gamma;
								double K = P1[j].pres/B + 1.0;
								rhoj = P2[j].rho_ref*pow(K, 1.0/k_gamma);
							}

							tmp_rhox+=-(rhoj/rhoj_ref-rhoi/rhoi_ref)*volj*tdwxc;
							tmp_rhoy+=-(rhoj/rhoj_ref-rhoi/rhoi_ref)*volj*tdwyc;
							
							// // Pressure based Autuono model
							// Real cc = k_soundspeed;
							// Real rho0=k_rho0_eos;

							// Real presj = P1[j].pres;
							// Real presi = P1[i].pres;
							
							// tmp_rhox+=-k_gamma/cc/cc/rho0*(presj-presi)*(mj/rhoj)*tdwxc;
							// tmp_rhoy+=-k_gamma/cc/cc/rho0*(presj-presi)*(mj/rhoj)*tdwyc;

						}

						// normal gradient for curvature
						if((k_fs_solve)&(k_surf_model==2))
						{
							// Real nC_s,nC_sx,nC_sy,nC_st;

							// nC_s=-(ptypei!=ptypej)*(ptypei*ptypej>0);
							// nC_st=nC_s*((mi/rhoi)*(mi/rhoi)+(mj/rhoj)*(mj/rhoj));
							// nC_st*=(rhoi/(rhoi+rhoj))*(rhoi/mi)*tmp_dwij;

							// nC_sx=nC_st*(xj-xi)/tdist;
							// nC_sy=nC_st*(yj-yi)/tdist;

							// tmp_nx+=nC_sx;
							// tmp_ny+=nC_sy;

							grad_lix+=(lj-li)*tdwxc*mj/rhoj*(P1[i].p_type==P1[j].p_type);
							grad_liy+=(lj-li)*tdwyc*mj/rhoj*(P1[i].p_type==P1[j].p_type);
						}

						if(k_fv_solve==Morris){
								
							tmp_uxx+=-volj*(uxi-uxj)*tdwx;
							tmp_uxy+=-volj*(uxi-uxj)*tdwy;
							tmp_uyx+=-volj*(uyi-uyj)*tdwx;
							tmp_uyy+=-volj*(uyi-uyj)*tdwy;
						}
						
						if(k_viscosity_type>0){
						shearxx += 2.0*(uxj-uxi)*tdwxc*mj/rhoj;
						shearxy += (uxj-uxi)*tdwyc*mj/rhoj+(uyj-uyi)*tdwxc*mj/rhoj;
						shearyx += (uyj-uyi)*tdwxc*mj/rhoj+(uxj-uxi)*tdwyc*mj/rhoj;
						shearyy += 2.0*(uyj-uyi)*tdwyc*mj/rhoj;
						}
					}
				}
			}
				}
			}
		}
	}


	// reference density
	// if(tcount==0){
		P2[i].rho_ref = 1000.0;
		if(P1[i].p_type==2)	P2[i].rho_ref = 1.0;

	// }	
	// filter
	if((tcount%k_freq_filt)==0) P1[i].flt_s=tmp_flt;

	// // normal gradient for surface tension
	// if((k_fs_solve)&(k_surf_model==2)){

	// 	P3[i].nx=tmp_nx;
	// 	P3[i].ny=tmp_ny;
	
	// 	Real ntmpnmg=sqrt(tmp_nx*tmp_nx+tmp_ny*tmp_ny);
	// 	P3[i].nmag=ntmpnmg;
	// 	if(ntmpnmg<1E-5){
	// 		P3[i].nx=0;
	// 		P3[i].ny=0;
	// 		P3[i].nmag=1e-20;
	// 	}
	// }

	Real gradli = sqrt(grad_lix*grad_lix+grad_liy*grad_liy);
	if(gradli>0.1*li/tmp_h){
		Real lambdamag = sqrt(grad_lix*grad_lix + grad_liy*grad_liy);
		P3[i].nx = -grad_lix/lambdamag;
		P3[i].ny = -grad_liy/lambdamag;
		P3[i].nmag = sqrt(P3[i].nx*P3[i].nx+P3[i].ny*P3[i].ny);
	}

	// gradient density
	if(k_delSPH_solve==Antuono){
		P1[i].grad_rhox=tmp_rhox;
		P1[i].grad_rhoy=tmp_rhoy;
	}

	if(k_fv_solve==Morris){

		P3[i].Gxx=tmp_uxx;
		P3[i].Gxy=tmp_uxy;
		P3[i].Gyx=tmp_uyx;
		P3[i].Gyy=tmp_uyy;
	}

	if(k_viscosity_type>0){
	P1[i].D = sqrt(0.5*(shearxx*shearxx + shearxy*shearxy + shearyx*shearyx + shearyy*shearyy));
	}

}

////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_prep3D(int_t*g_str,int_t*g_end,part1*P1, part2*P2, part3*P3, int_t tcount)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part3) return;
	if(P1[i].i_type>i_type_crt) return;

	int_t icell,jcell,kcell;
	int_t ptypei=P1[i].p_type;

	Real xi,yi,zi;
	Real uxi,uyi,uzi;
	Real mi=P1[i].m;
	Real rhoi=P1[i].rho;
	Real cci=P3[i].cc;
	Real tmp_h,tmp_A,search_range;
	Real tmp_flt,tmp_SR;
	Real tmp_rhox,tmp_rhoy,tmp_rhoz;
	Real tmp_ncx, tmp_ncy, tmp_ncz, tmp_nx, tmp_ny, tmp_nz;
	Real tvis_t=0.0,th;
	Real li = P3[i].lambda;
	Real grad_lix, grad_liy, grad_liz;

	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;

	uxi=P1[i].ux;
	uyi=P1[i].uy;
	uzi=P1[i].uz;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	th=tmp_h*L_SPS;
	search_range=k_search_kappa*tmp_h;	// search range

	// calculate I,J,K in cell
if((k_x_max==k_x_min)){icell=0;}
else{icell=min(floor((xi-k_x_min)/k_dcell),k_NI-1);}
if((k_y_max==k_y_min)){jcell=0;}
else{jcell=min(floor((yi-k_y_min)/k_dcell),k_NJ-1);}
if((k_z_max==k_z_min)){kcell=0;}
else{kcell=min(floor((zi-k_z_min)/k_dcell),k_NK-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

	// 초기화
	tmp_flt=tmp_SR=tmp_rhox=tmp_rhoy=tmp_rhoz=0.0;
	tmp_nx=tmp_ny=tmp_nz=tmp_ncx=tmp_ncy=tmp_ncz=0.0;
	grad_lix=grad_liy=grad_liz=0.0;


	// 계산
	for(int_t z=-P1[i].ncell;z<=P1[i].ncell;z++){
		for(int_t y=-P1[i].ncell;y<=P1[i].ncell;y++){
			for(int_t x=-P1[i].ncell;x<=P1[i].ncell;x++){
				// int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				int_t k=idx_cell(icell+x,jcell+y,kcell+z);

			//	if(k<0||k>=k_num_cells-1) continue;
				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;
				if(g_str[k]!=cu_memset){
					int_t fend=g_end[k];
					for(int_t j=g_str[k];j<fend;j++){
						Real xj,yj,zj,uxj,uyj,uzj,uij2,mj,rhoj,ccj, tdwx,tdwy,tdwz,tmp_wij,tmp_dwij,tdist,tmp_val;
						int_t ptypej;

						xj=P1[j].x;
						yj=P1[j].y;
						zj=P1[j].z;
						uxj=P1[j].ux;
						uyj=P1[j].uy;
						uzj=P1[j].uz;
						mj=P1[j].m;
						rhoj=P1[j].rho;
						ptypej=P1[j].p_type;
						ccj=P3[j].cc;
						Real lj = P3[j].lambda;

						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj))+1e-20;

						if(tdist<search_range){
							tmp_wij=calc_kernel_wij(tmp_A,tmp_h,tdist);
							tmp_dwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);

							tdwx=tmp_dwij*(xi-xj)/tdist;
							tdwy=tmp_dwij*(yi-yj)/tdist;
							tdwz=tmp_dwij*(zi-zj)/tdist;

							// filter
							if((tcount%k_freq_filt)==0) tmp_flt+=mj/rhoj*tmp_wij;

							// strain rate
							if((k_fv_solve==1)&&(k_turbulence_model!=Laminar))
							{
								uij2=(uxi-uxj)*(uxi-uxj)+(uyi-uyj)*(uyi-uyj)+(uzi-uzj)*(uzi-uzj);
								tmp_val=-0.5*mj*(rhoi+rhoj)*uij2;
								tmp_val/=(rhoi*rhoj*tdist*tdist);
								tmp_SR+=tmp_val*(xi-xj)*tdwx+tmp_val*(yi-yj)*tdwy+tmp_val*(zi-zj)*tdwz;
							}

							// gradient rho (for delta-sph)
							if(k_delSPH_solve==Antuono)
							{
								apply_gradient_correction_3D(P3[i].Cm,tmp_wij,tdwx,tdwy,tdwz,&tdwx,&tdwy,&tdwz);

								// tmp_rhox+=-(rhoj-rhoi)*(mj/rhoj)*tdwx;
								// tmp_rhoy+=-(rhoj-rhoi)*(mj/rhoj)*tdwy;
								// tmp_rhoz+=-(rhoj-rhoi)*(mj/rhoj)*tdwz;

								Real rhoi_ref=P2[i].rho_ref;
								Real rhoj_ref=P2[j].rho_ref;
	
								tmp_rhox+=-(rhoj/rhoj_ref-rhoi/rhoi_ref)*(mj/rhoj)*tdwx;
								tmp_rhoy+=-(rhoj/rhoj_ref-rhoi/rhoi_ref)*(mj/rhoj)*tdwy;
								tmp_rhoz+=-(rhoj/rhoj_ref-rhoi/rhoi_ref)*(mj/rhoj)*tdwz;
							}

							// normal gradient for curvature
							if((k_fs_solve)&(k_surf_model==2))
							{
								Real tdwxc=P3[i].Cm[0][0]*tdwx+P3[i].Cm[0][1]*tdwy+P3[i].Cm[0][2]*tdwz;
								Real tdwyc=P3[i].Cm[1][0]*tdwx+P3[i].Cm[1][1]*tdwy+P3[i].Cm[1][2]*tdwz;
								Real tdwzc=P3[i].Cm[2][0]*tdwx+P3[i].Cm[2][1]*tdwy+P3[i].Cm[2][2]*tdwz;

								if(li>=0.7){
									grad_lix+=(lj-li)*tdwxc*mj/rhoj;
									grad_liy+=(lj-li)*tdwyc*mj/rhoj;
									grad_liz+=(lj-li)*tdwzc*mj/rhoj;
								}else{
									grad_lix+=(lj)*tdwxc*mj/rhoj;
									grad_liy+=(lj)*tdwyc*mj/rhoj;
									grad_liz+=(lj)*tdwzc*mj/rhoj;
								}

								// Real nC_s,nC_sx,nC_sy,nC_sz,nC_st;

								// nC_s=(ptypei!=ptypej)*(ptypei*ptypej>0);
								// nC_st=nC_s*((mi/rhoi)*(mi/rhoi)+(mj/rhoj)*(mj/rhoj));
								// nC_st*=(rhoi/(rhoi+rhoj))*(rhoi/mi)*tmp_dwij;

								// nC_sx=nC_st*(xj-xi)/tdist;
								// nC_sy=nC_st*(yj-yi)/tdist;
								// nC_sz=nC_st*(zj-zi)/tdist;

								// tmp_nx+=nC_sx;
								// tmp_ny+=nC_sy;
								// tmp_nz+=nC_sz;


							}
						}
					}
				}
			}
		}
	}

	// strain_rate
	if((k_fv_solve==1)&&(k_turbulence_model!=Laminar)) {
		tmp_SR=max(1e-20,tmp_SR);
		P2[i].SR=sqrt(tmp_SR);

		if(k_turbulence_model==SPS) tvis_t=(Cs_SPS*th)*(Cs_SPS*th)*tmp_SR;
		P3[i].vis_t=tvis_t*rhoi;
	}

	// reference density
	if(tcount==0){
	P2[i].rho_ref=P1[i].rho;
	}

	// filter
	if((tcount%k_freq_filt)==0) P1[i].flt_s=tmp_flt;

	// // switch p_type
	// if(k_switch_ptype==1) P1[i].p_type=functions_switch_ptype();

	// gradient density
	if(k_delSPH_solve==Antuono){
		P1[i].grad_rhox=tmp_rhox;
		P1[i].grad_rhoy=tmp_rhoy;
		P1[i].grad_rhoz=tmp_rhoz;
	}

	// normal gradient for surface tension
	if((k_fs_solve)&(k_surf_model==2)){

	// 	// KERNEL_clc_normal_gradient3D ---------------
	// 	P3[i].nx=tmp_nx;
	// 	P3[i].ny=tmp_ny;
	// 	P3[i].nz=tmp_nz;

	// 	Real ntmpnmg=sqrt(tmp_nx*tmp_nx+tmp_ny*tmp_ny+tmp_nz*tmp_nz);
	// 	P3[i].nmag=ntmpnmg;
	// 	if(ntmpnmg<NORMAL_THRESHOLD){
	// 		P3[i].nx=0;
	// 		P3[i].ny=0;
	// 		P3[i].nz=0;
	// 		P3[i].nmag=1e-20;
	// 	}

	Real gradli = sqrt(grad_lix*grad_lix+grad_liy*grad_liy+grad_liz*grad_liz);
		if(gradli>0.1*li/tmp_h){
			Real lambdamag = sqrt(grad_lix*grad_lix + grad_liy*grad_liy + grad_liz*grad_liz);
			P3[i].nx = -grad_lix/lambdamag;
			P3[i].ny = -grad_liy/lambdamag;
			P3[i].nz = -grad_liz/lambdamag;
			P3[i].nmag = sqrt(P3[i].nx*P3[i].nx+P3[i].ny*P3[i].ny+P3[i].nz*P3[i].nz);
		}
	}
}

////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_correction_KGC_2D(int_t*g_str,int_t*g_end,part1*P1,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	P3[i].lambda=0;
	if(i>=k_num_part3) return;
	if(P1[i].i_type>i_type_crt) return;
	// if(P1[i].p_type==0||P1[i].p_type==2)	return;

	int_t icell,jcell;
	Real search_range,tmp_h,tmp_A;
	Real xi,yi;
	Real tmpxx,tmpyy,tmpxy;
	Real tmpxx2,tmpyy2,tmpxy2;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	xi=P1[i].x;
	yi=P1[i].y;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;

	tmpxx=tmpyy=tmpxy=0;
	tmpxx2=tmpyy2=tmpxy2=0;

	for(int_t y=-P1[i].ncell;y<=P1[i].ncell;y++){
		for(int_t x=-P1[i].ncell;x<=P1[i].ncell;x++){
			//int_t k=(icell+x)+k_NI*(jcell+y);
			int_t k=idx_cell(icell+x,jcell+y,0);
			if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))) continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){

					Real xj,yj,tdist;
					xj=P1[j].x;
					yj=P1[j].y;

					tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj));
					Real tmp_hij;
					if(k_multi_type==3||k_multi_type==4)	tmp_hij=(P1[i].h+P1[j].h)/2;
					else	tmp_hij=P1[i].h;
					Real tmp_Aij=calc_tmpA(tmp_hij);

					search_range=k_search_kappa*tmp_hij;
					if(tdist>0&&tdist<search_range){
						Real tdwij,mj,rhoj,txx,txy,tyy,rtd,mtd;
						Real txx2,txy2,tyy2;
						tdwij=calc_kernel_dwij(tmp_Aij,tmp_hij,tdist);
						mj=P1[j].m;
						rhoj=P1[j].rho;
						mtd=mj*tdwij;
						rtd=1.0/(rhoj*tdist);

						txx=-mtd*(xi-xj)*(xi-xj);
						txx*=rtd;
						txy=-mtd*(yi-yj)*(xi-xj);
						txy*=rtd;
						tyy=-mtd*(yi-yj)*(yi-yj);
						tyy*=rtd;

						txx2=-mtd*(xi-xj)*(xi-xj)*((P1[i].p_type==P1[j].p_type)||(P1[j].p_type<=0));
						txx2*=rtd;
						txy2=-mtd*(yi-yj)*(xi-xj)*((P1[i].p_type==P1[j].p_type)||(P1[j].p_type<=0));
						txy2*=rtd;
						tyy2=-mtd*(yi-yj)*(yi-yj)*((P1[i].p_type==P1[j].p_type)||(P1[j].p_type<=0));
						tyy2*=rtd;

						tmpxx+=txx;
						tmpxy+=txy;
						tmpyy+=tyy;

						tmpxx2+=txx2;
						tmpxy2+=txy2;
						tmpyy2+=tyy2;
					}
				}
			// }
			}
		}
	}
	// save values to particle array

	Real tmpcmd=tmpxx*tmpyy-tmpxy*tmpxy;
	if(abs(tmpcmd)>Min_det){
		Real rtcmd=1.0/tmpcmd;
		P3[i].Cm[0][0]=tmpyy*rtcmd;
		P3[i].Cm[0][1]=-tmpxy*rtcmd;
		P3[i].Cm[1][0]=-tmpxy*rtcmd;
		P3[i].Cm[1][1]=tmpxx*rtcmd;
	}else{
		P3[i].Cm[0][0]=1;
		P3[i].Cm[0][1]=0;
		P3[i].Cm[1][0]=0;
		P3[i].Cm[1][1]=1;
	}

	Real tmpcmd2=tmpxx2*tmpyy2-tmpxy2*tmpxy2;
	if(abs(tmpcmd2)>Min_det){
		Real rtcmd2=1.0/tmpcmd2;
		P3[i].lambda = 1.0/max(tmpyy2*rtcmd2,tmpxx2*rtcmd2);
	}else{
		P3[i].lambda = 1.0;
	}

}

////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_correction_KGC_3D(int_t*g_str,int_t*g_end,part1*P1,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part3) return;
	if(P1[i].i_type>i_type_crt) return;

	int_t icell,jcell,kcell;
	Real xi,yi,zi;
	Real search_range,tmp_h,tmp_A;;
	Real tmpxx,tmpyy,tmpzz,tmpxy,tmpyz,tmpzx;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	if((k_z_max==k_z_min)){kcell=0;}
	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

	tmpxx=tmpyy=tmpzz=0;
	tmpxy=tmpyz=tmpzx=0;
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

						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));
						if(tdist>0&&tdist<search_range){
							Real tdwij,mj,rhoj,txx,txy,tyy,tzx,tyz,tzz,rtd,mtd;
							tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);
							mj=P1[j].m;
							rhoj=P1[j].rho;

							mtd=mj*tdwij;
							rtd=1.0/(rhoj*tdist);

							txx=-mtd*(xi-xj)*(xi-xj);
							txx*=rtd;
							txy=-mtd*(yi-yj)*(xi-xj);
							txy*=rtd;
							tyy=-mtd*(yi-yj)*(yi-yj);
							tyy*=rtd;
							tzx=-mtd*(xi-xj)*(zi-zj);
							tzx*=rtd;
							tyz=-mtd*(yi-yj)*(zi-zj);
							tyz*=rtd;
							tzz=-mtd*(zi-zj)*(zi-zj);
							tzz*=rtd;
							tmpxx+=txx;
							tmpxy+=txy;
							tmpyy+=tyy;
							tmpzx+=tzx;
							tmpyz+=tyz;
							tmpzz+=tzz;
						}
					}
				}
			}
		}
	}
	// save values to particle array
	Real tmpcmd;
	tmpcmd=tmpxx*(tmpyy*tmpzz-tmpyz*tmpyz);
	tmpcmd-=tmpxy*(tmpxy*tmpzz-tmpyz*tmpzx);
	tmpcmd+=tmpzx*(tmpxy*tmpyz-tmpyy*tmpzx);

	if(abs(tmpcmd)>Min_det){
		Real rtcmd=1.0/tmpcmd;
		P3[i].Cm[0][0]=(tmpyy*tmpzz-tmpyz*tmpyz)*rtcmd;
		P3[i].Cm[0][1]=(tmpzx*tmpyz-tmpxy*tmpzz)*rtcmd;
		P3[i].Cm[0][2]=(tmpxy*tmpyz-tmpzx*tmpyy)*rtcmd;
		P3[i].Cm[1][0]=(tmpzx*tmpyz-tmpxy*tmpzz)*rtcmd;
		P3[i].Cm[1][1]=(tmpxx*tmpzz-tmpzx*tmpzx)*rtcmd;
		P3[i].Cm[1][2]=(tmpzx*tmpxy-tmpxx*tmpyz)*rtcmd;
		P3[i].Cm[2][0]=(tmpxy*tmpyz-tmpzx*tmpyy)*rtcmd;
		P3[i].Cm[2][1]=(tmpzx*tmpxy-tmpxx*tmpyz)*rtcmd;
		P3[i].Cm[2][2]=(tmpxx*tmpyy-tmpxy*tmpxy)*rtcmd;
	}
	else{
		P3[i].Cm[0][0]=1;
		P3[i].Cm[0][1]=0;
		P3[i].Cm[0][2]=0;
		P3[i].Cm[1][0]=0;
		P3[i].Cm[1][1]=1;
		P3[i].Cm[1][2]=0;
		P3[i].Cm[2][0]=0;
		P3[i].Cm[2][1]=0;
		P3[i].Cm[2][2]=1;
	}

	P3[i].lambda = 1.0/max(max(P3[i].Cm[0][0],P3[i].Cm[1][1]),P3[i].Cm[2][2]);

}

void gradient_correction(int_t*g_str,int_t*g_end,part1*P1,part3*P3)
{
	dim3 b,t;
	t.x=128;
	b.x=(num_part3-1)/t.x+1;
	if(dim==2) KERNEL_clc_correction_KGC_2D<<<b,t>>>(g_str,g_end,P1,P3);
	if(dim==3) KERNEL_clc_correction_KGC_3D<<<b,t>>>(g_str,g_end,P1,P3);
}

////////////////////////////////////////////////////////////////////////
// calcuate color field for two-phase flow surface tension model (2017.05.08 jyb)
__global__ void KERNEL_clc_color_field2D(int_t*g_str,int_t*g_end,part1*P1,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part3) return;
	if(P1[i].i_type>i_type_crt) return;

	int_t ptypei;
	int_t icell,jcell;
	Real xi,yi;
	Real tmpn,tmpd;
	Real search_range,tmp_h,tmp_A;

	ptypei=P1[i].p_type;
	xi=P1[i].x;
	yi=P1[i].y;
	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;

	tmpn=tmpd=0.0;

	for(int_t y=-P1[i].ncell;y<=P1[i].ncell;y++){
		for(int_t x=-P1[i].ncell;x<=P1[i].ncell;x++){
			// int_t k=(icell+x)+k_NI*(jcell+y);
			int_t k=idx_cell(icell+x,jcell+y,0);

			if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))) continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					Real xj,yj,tdist; //rr
					xj=P1[j].x;
					yj=P1[j].y;

					tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj));
					if(tdist<search_range){
						Real twij,mj,rhoj;
						twij=calc_kernel_wij(tmp_A,tmp_h,tdist);

						mj=P1[j].m;
						rhoj=P1[j].rho;

						tmpn+=mj*twij*(ptypei==P1[j].p_type)/rhoj;
						tmpd+=mj*twij/rhoj;
					}
				}
			}
		}
	}
	P3[i].cc=tmpn/tmpd;
}
////////////////////////////////////////////////////////////////////////
// calcuate color field for two-phase flow surface tension model (2017.05.08 jyb)
__global__ void KERNEL_clc_color_field3D(int_t*g_str,int_t*g_end,part1*P1,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part3) return;
	if(P1[i].i_type>i_type_crt) return;


	int_t ptypei;
	int_t icell,jcell,kcell;
	Real xi,yi,zi;
	Real tmpn,tmpd;
	Real search_range,tmp_h,tmp_A;

	ptypei=P1[i].p_type;
	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;

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

	tmpn=tmpd=0.0;

	for(int_t z=-P1[i].ncell;z<=P1[i].ncell;z++){
		for(int_t y=-P1[i].ncell;y<=P1[i].ncell;y++){
			for(int_t x=-P1[i].ncell;x<=P1[i].ncell;x++){
				//int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				int_t k=idx_cell(icell+x,jcell+y,kcell+z);

				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;
				if(g_str[k]!=cu_memset){
					int_t fend=g_end[k];
					for(int_t j=g_str[k];j<fend;j++){
						Real xj,yj,zj,tdist; //rr
						xj=P1[j].x;
						yj=P1[j].y;
						zj=P1[j].z;

						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));

						if(tdist<search_range){
							Real twij,mj,rhoj;
							twij=calc_kernel_wij(tmp_A,tmp_h,tdist);

							mj=P1[j].m;
							rhoj=P1[j].rho;

							tmpn+=mj*twij*(ptypei==P1[j].p_type)/rhoj;
							tmpd+=mj*twij/rhoj;
						}
					}
				}
			}
		}
	}
	P3[i].cc=tmpn/tmpd;
}


//////////////////////////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_correction_preLaplacian(int_t*g_str,int_t*g_end,part1*P1,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part3) return;
	if(P1[i].i_type>i_type_crt) return;

	int_t icell,jcell;
	Real xi, yi;
	Real search_range,tmp_h,tmp_A;
	Real tA111, tA112, tA122, tA211, tA212, tA222;
	Real tflt_s;
	
	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	xi=P1[i].x;
	yi=P1[i].y;

	tA111=tA112=tA122=tA211=tA212=tA222=0.0;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;

	for(int_t y=-P1[i].ncell;y<=P1[i].ncell;y++){
		for(int_t x=-P1[i].ncell;x<=P1[i].ncell;x++){
			//int_t k=(icell+x)+k_NI*(jcell+y);
			int_t k=idx_cell(icell+x,jcell+y,0);

			if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))) continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					if(P1[j].p_type<1000){

					Real xj,yj,tdist;

					xj=P1[j].x;
					yj=P1[j].y;

					tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj));

					if(tdist>0&&tdist<search_range){
						Real twij,mj,rhoj,rt,mtd;
						Real tdwij, tdwx, tdwy;
						Real tmprho;

						twij=calc_kernel_wij(tmp_A,tmp_h,tdist);
						tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);
						
						apply_gradient_correction_2D(P3[i].Cm,twij,tdwx,tdwy,&tdwx,&tdwy);

						tdwx=tdwij*(xi-xj)/tdist;
						tdwy=tdwij*(yi-yj)/tdist;

						mj=P1[j].m;
						rhoj=P1[j].rho;
						tmprho=mj/rhoj;
						
						tA111+=(xi-xj)*(xi-xj)*tdwx*tmprho;
						tA112+=(xi-xj)*(yi-yj)*tdwx*tmprho;
						tA122+=(yi-yj)*(yi-yj)*tdwx*tmprho;
						tA211+=(xi-xj)*(xi-xj)*tdwy*tmprho;
						tA212+=(xi-xj)*(yi-yj)*tdwy*tmprho;
						tA222+=(yi-yj)*(yi-yj)*tdwy*tmprho;
					
					}
				}
			}
			}
		}
	}
	P3[i].A111=tA111;
	P3[i].A112=tA112;
	P3[i].A122=tA122;
	P3[i].A211=tA211;
	P3[i].A212=tA212;
	P3[i].A222=tA222;

}

//////////////////////////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_correction_Laplacian(int_t*g_str,int_t*g_end,part1*P1,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part3) return;
	if(P1[i].i_type>i_type_crt) return;

	int_t icell,jcell;
	Real xi, yi;
	Real search_range,tmp_h,tmp_A;
	Real tB11, tB12, tB13, tB21, tB22, tB23, tB31, tB32, tB33;
	Real tA111, tA112, tA122, tA211, tA212, tA222; 
	Real tflt_s;
	
	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	xi=P1[i].x;
	yi=P1[i].y;

	tB11=tB12=tB13=tB21=tB22=tB23=tB31=tB32=tB33=0.0;
	
	tA111=P3[i].A111;
	tA112=P3[i].A112;
	tA122=P3[i].A122;
	tA211=P3[i].A211;
	tA212=P3[i].A212;
	tA222=P3[i].A222;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;

	for(int_t y=-P1[i].ncell;y<=P1[i].ncell;y++){
		for(int_t x=-P1[i].ncell;x<=P1[i].ncell;x++){
			//int_t k=(icell+x)+k_NI*(jcell+y);
			int_t k=idx_cell(icell+x,jcell+y,0);

			if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))) continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					if(P1[j].p_type<1000){

					Real xj,yj,tdist;
					xj=P1[j].x;
					yj=P1[j].y;

					tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj));
					if(tdist>0&&tdist<search_range){
						Real twij,mj,rhoj,rt,mtd;
						Real tdwij, tdwx, tdwy;
						Real tmprho;
						Real e1,e2,r1,r2;

						twij=calc_kernel_wij(tmp_A,tmp_h,tdist);
						tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);
						
						tdwx=tdwij*(xi-xj)/tdist;
						tdwy=tdwij*(yi-yj)/tdist;

						e1=(xi-xj)/tdist;
						e2=(yi-yj)/tdist;

						r1=(xi-xj);
						r2=(yi-yj);

						mj=P1[j].m;
						rhoj=P1[j].rho;
						tmprho=mj/rhoj;

						tB11+=tmprho*(tA111*e1+tA211*e2+r1*e1)*(e1*tdwx);
						tB12+=tmprho*(tA111*e1+tA211*e2+r1*e1)*(e1*tdwy+e2*tdwx);
						tB13+=tmprho*(tA111*e1+tA211*e2+r1*e1)*(e2*tdwy);
						tB21+=tmprho*(tA112*e1+tA212*e2+r1*e2)*(e1*tdwx);
						tB22+=tmprho*(tA112*e1+tA212*e2+r1*e2)*(e1*tdwy+e2*tdwx);
						tB23+=tmprho*(tA112*e1+tA212*e2+r1*e2)*(e2*tdwy);
						tB31+=tmprho*(tA122*e1+tA222*e2+r2*e2)*(e1*tdwx);
						tB32+=tmprho*(tA122*e1+tA222*e2+r2*e2)*(e1*tdwy+e2*tdwx);
						tB33+=tmprho*(tA122*e1+tA222*e2+r2*e2)*(e2*tdwy);
					}
				}
			}
			}
		}
	}

	Real det_B;
	Real tL11, tL12, tL22;
	tL11=tL12=tL22=0.0;
	
	det_B=tB11*(tB22*tB33-tB23*tB32)-tB12*(tB21*tB33-tB23*tB31)+tB13*(tB21*tB32-tB22*tB31);

	if(det_B>1e-10){
		tL11=-1.0/det_B*(tB22*tB33-tB23*tB32+tB12*tB23-tB13*tB22);
		tL12=-1.0/det_B*(-tB21*tB33+tB23*tB31-tB11*tB23+tB13*tB21);
		tL22=-1.0/det_B*(tB21*tB32-tB22*tB31+tB11*tB22-tB12*tB21);
	}
	else{
	 	tL11=1.0;
	 	tL12=0.0;
	 	tL22=1.0;
	}


	P3[i].L11=tL11;
	P3[i].L12=tL12;
	P3[i].L21=tL12;
	P3[i].L22=tL22;

}

// Preparation step before the main solver: applies gradient correction and pre-processing kernels
void preparation(
    dim3 b, dim3 t,
    int_t* g_str, int_t* g_end,
    part1* dev_SP1, part2* dev_SP2, part3* dev_P3
) {
    // Apply gradient correction (host or device function)
    gradient_correction(g_str, g_end, dev_SP1, dev_P3);
    cudaDeviceSynchronize();

    // Apply pre-processing kernel depending on dimension
    if (dim == 2)
        KERNEL_clc_prep2D<<<b, t>>>(g_str, g_end, dev_SP1, dev_SP2, dev_P3, count);
    if (dim == 3)
        KERNEL_clc_prep3D<<<b, t>>>(g_str, g_end, dev_SP1, dev_SP2, dev_P3, count);

    cudaDeviceSynchronize();
}
