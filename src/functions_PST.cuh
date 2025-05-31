////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_surface_detect2D(int_t*g_str,int_t*g_end,part1*P1,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	P3[i].lbl_surf=0;
	if(i>=k_num_part3) return;
	if(P1[i].p_type>=1000)	return;		// Immersed Boundary Method
	if(P1[i].p_type<=0)	return;		// Immersed Boundary Method

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
							Real tmp_hij;
							if(k_multi_type==3||k_multi_type==4)	tmp_hij=(P1[i].h+P1[j].h)/2;
							else	tmp_hij=P1[i].h;
							Real tmp_Aij=calc_tmpA(tmp_hij);
		
							search_range=k_search_kappa*tmp_hij;
					if(tdist<search_range){
						Real rd=1.0/(tdist+1e-10);
						Real tdwij=calc_kernel_dwij(tmp_Aij,tmp_hij,tdist);
						Real tdwx, tdwy;
						Real rhoj, mj;
						rhoj = P1[j].rho;
						mj = P1[j].m;

						tdwx=tdwij*(xi-xj)/(tdist+1e-10);
						tdwy=tdwij*(yi-yj)/(tdist+1e-10);

						int_t flag = ((ptypei==ptypej));
						tmpx+=-(xi-xj)*tdwx*mj/rhoj*flag;
						tmpy+=-(yi-yj)*tdwy*mj/rhoj*flag;
						non++;
					}
				}
			}
			}
			}
		}
	}
	Real detect = tmpx+tmpy;
	if(detect<1.5)	P3[i].lbl_surf=2;
	if((detect>=1.5)&&(detect<2.0))	P3[i].lbl_surf=1;
	if(detect>=2.0)	P3[i].lbl_surf=0;
	if(non<10)	P3[i].lbl_surf=3;
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_surface_detect3D(int_t inout,int_t*g_str,int_t*g_end,part1*P1,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part3) return;
	if(P1[i].i_type!=inout) return;

	int_t non;
	int_t icell,jcell,kcell;
	Real xi,yi,zi;
	Real rn;
	Real tmpx,tmpy,tmpz;
	Real search_range;

	search_range=k_search_kappa*P1[i].h;

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

	tmpx=tmpy=tmpz=0.0;
	non=0;
	for(int_t z=-P1[i].ncell;z<=P1[i].ncell;z++){
		for(int_t y=-P1[i].ncell;y<=P1[i].ncell;y++){
			for(int_t x=-P1[i].ncell;x<=P1[i].ncell;x++){
				int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				if(k<0||k>=k_num_cells-1) continue;
				if(g_str[k]!=cu_memset){
					int_t fend=g_end[k];
					for(int_t j=g_str[k];j<fend;j++){
						Real xj,yj,zj,rr,tdist;
						xj=P1[j].x;
						yj=P1[j].y;
						zj=P1[j].z;

						rr=(xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj);
						tdist=sqrt(rr);
						if(tdist<search_range){
							Real rd=1.0/(tdist+1e-10);
							tmpx+=(xi-xj)*rd;
							tmpy+=(yi-yj)*rd;
							tmpz+=(zi-zj)*rd;
							non++;
						}
					}
				}
			}
		}
	}
	rn=1.0/non;
	tmpx=tmpx*rn;
	tmpy=tmpy*rn;
	tmpz=tmpz*rn;
	P3[i].lbl_surf=((sqrt(tmpx*tmpx+tmpy*tmpy+tmpz+tmpz))>0.3)|(non<10);
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_freesurface_normalvector(int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3,Real tdt)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part3) return;
	if(P1[i].i_type==3) return;
	if(P1[i].p_type==9) return;
	if(P1[i].p_type<=0) return;
	if(P3[i].lbl_surf==0)	return;
	if(P1[i].buffer_type>0) return;
	if((P1[i].elix<1e-10)&(P1[i].eliy<1e-10)) return;
	if(P1[i].p_type>=1000)	return;		// Immersed Boundary Method

	int_t ptypei;
	int_t icell,jcell;
	Real xi,yi,uxi,uyi,rhoi,mi;
	Real w_dx_i,dr_square,hi;
	Real tmpx,tmpy,tmprhox,tmprhoy;
	Real tmpuxx, tmpuxy, tmpuyx, tmpuyy;
	Real tmpmx,tmpmy;
	Real conc;
	Real concx, concy;

	Real search_range,tmp_A;

	hi=P1[i].h;
	tmp_A=calc_tmpA(hi);
	search_range=k_search_kappa*hi;	// search range

	ptypei=P1[i].p_type;

	xi=P1[i].x;
	yi=P1[i].y;
	uxi=P1[i].ux;
	uyi=P1[i].uy;
	w_dx_i=P1[i].w_dx;
	rhoi=P1[i].rho;
	mi=P1[i].m;


	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;

	tmpx=tmpy=0.0;
	tmprhox=tmprhoy=0.0;
	tmpuxx=tmpuxy=tmpuyx=tmpuyy=0.0;
	tmpmx=tmpmy=0.0;
	conc=0.0;
	concx=concy=0.0;

	for(int_t y=-P1[i].ncell;y<=P1[i].ncell;y++){
		for(int_t x=-P1[i].ncell;x<=P1[i].ncell;x++){
			int_t k=(icell+x)+k_NI*(jcell+y);
			if(k<0||k>=k_num_cells-1)	continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					Real xj,yj,tdist,elixj,eliyj;
					Real uxj, uyj;
					int_t ptypej;

					xj=P1[j].x;
					yj=P1[j].y;

					uxj=P1[j].ux;
					uyj=P1[j].ux;
					elixj=P1[j].elix;
					eliyj=P1[j].eliy;
					ptypej=P1[j].p_type;

					// if((elixj>1e-10)||(eliyj>1e-10)||(ptypej!=1)){
						if(P1[j].p_type<1000){
					Real tmp_hij;
					if(k_multi_type==3||k_multi_type==4)	tmp_hij=(P1[i].h+P1[j].h)/2;
					else	tmp_hij=P1[i].h;
					Real tmp_Aij=calc_tmpA(tmp_hij);

					search_range=k_search_kappa*tmp_hij;
					tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+1E-20);
					if(tdist<search_range){
						Real tdwx,tdwy,tdwij,twij,hj,mj,rhoj,ww,volj;
						tdwij=calc_kernel_dwij(tmp_Aij,tmp_hij,tdist);
						twij=calc_kernel_wij(tmp_Aij,tmp_hij,tdist);

						tdwx=tdwij*(xi-xj)/tdist;
						tdwy=tdwij*(yi-yj)/tdist;
						
						// apply_gradient_correction_2D(P3[i].Cm,twij,tdwx,tdwy,&tdwx,&tdwy);

						mj=P1[j].m;
						hj=P1[j].h;
						rhoj=P1[j].rho;
						volj=P1[j].vol;

						int_t flag = ((ptypei==ptypej)||(ptypej<=0));

						concx+=mj/rhoj*tdwx*flag;
						concy+=mj/rhoj*tdwy*flag;

					}
				}
			}
		}
		}
	}


	Real concdiff = sqrt(concx*concx+concy*concy);
	Real nx = -concx/concdiff;
	Real ny = -concy/concdiff;

	// if((P3[i].lbl_surf==1)||(P3[i].lbl_surf==2)){
	if((P3[i].lambda<0.85)){
		P3[i].nx_s = nx;
		P3[i].ny_s = ny;
		P3[i].tx_s = -ny;
		P3[i].ty_s = nx;
	}else{
		P3[i].nx_s = 0.0;
		P3[i].ny_s = 0.0;
		P3[i].tx_s = 0.0;
		P3[i].ty_s = 0.0;
	}
}
////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_particle_shifting_lind2D(int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3,Real tdt)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part3) return;
	if(P1[i].i_type==3) return;
	if(P1[i].p_type==9) return;
	if(P1[i].p_type<=0) return;
	if(P1[i].buffer_type>0) return;
	if((P1[i].elix<1e-10)&(P1[i].eliy<1e-10)) return;
	if(P1[i].p_type>=1000)	return;		// Immersed Boundary Method
	if(P1[i].p_type!=1)	return;		// Immersed Boundary Method

	int_t ptypei;
	int_t icell,jcell;
	Real xi,yi,uxi,uyi,rhoi,mi,Hi,Ti,presi;
	Real w_dx_i,dr_square,hi;
	Real tmpx,tmpy,tmprhox,tmprhoy,tmppresx,tmppresy;
	Real tmpuxx, tmpuxy, tmpuyx, tmpuyy;
	Real tmpmx,tmpmy,tmphx,tmphy,tmptx,tmpty;
	Real flt;
	Real conc;

	Real search_range,tmp_A;

	hi=P1[i].h;

	ptypei=P1[i].p_type;

	xi=P1[i].x;
	yi=P1[i].y;
	uxi=P1[i].ux;
	uyi=P1[i].uy;
	rhoi=P1[i].rho;
	presi=P1[i].pres;
	mi=P1[i].m;
	Hi=P1[i].enthalpy;
	Ti=P1[i].temp;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;

	tmpx=tmpy=0.0;
	tmprhox=tmprhoy=0.0;
	tmppresx=tmppresy=0.0;
	tmpuxx=tmpuxy=tmpuyx=tmpuyy=0.0;
	tmpmx=tmpmy=tmphx=tmphy=tmptx=tmpty=0.0;
	conc=0.0;
	flt=0.0;

	for(int_t y=-P1[i].ncell;y<=P1[i].ncell;y++){
		for(int_t x=-P1[i].ncell;x<=P1[i].ncell;x++){
			int_t k=(icell+x)+k_NI*(jcell+y);
			if(k<0||k>=k_num_cells-1)	continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					Real xj,yj,tdist,elixj,eliyj,hj;
					Real uxj, uyj;
					int_t ptypej;

					xj=P1[j].x;
					yj=P1[j].y;
					hj=P1[j].h;

					Real hshift = hi;

					uxj=P1[j].ux;
					uyj=P1[j].uy;
					elixj=P1[j].elix;
					eliyj=P1[j].eliy;
					ptypej=P1[j].p_type;
					search_range=k_search_kappa*hshift;	// search range

						if(P1[j].p_type!=2){
					Real tmp_hij;
					if(k_multi_type==3||k_multi_type==4)	tmp_hij=(P1[i].h+P1[j].h)/2;
					else	tmp_hij=P1[i].h;
					Real tmp_Aij=calc_tmpA(tmp_hij);

					search_range=k_search_kappa*tmp_hij;
					tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+1E-20);
					if(tdist<search_range){
						Real tdwx,tdwy,tdwij,twij,hj,mj,rhoj,presj,ww,volj,Hj,Tj;
						Real D_lind;

						tdwij=calc_kernel_dwij(tmp_Aij,tmp_hij,tdist);
						twij=calc_kernel_wij(tmp_Aij,tmp_hij,tdist);
						w_dx_i=P1[i].w_dx;

						tdwx=tdwij*(xi-xj)/tdist;
						tdwy=tdwij*(yi-yj)/tdist;
						
						// apply_gradient_correction_2D(P3[i].Cm,twij,tdwx,tdwy,&tdwx,&tdwy);

						mj=P1[j].m;
						hj=P1[j].h;
						rhoj=P1[j].rho;
						presj=P1[j].pres;
						// if(P1[j].p_type<=0)	presj -= rhoi*P1[j].dpres;

						volj=P1[j].vol;
						Hj=P1[j].enthalpy;
						Tj=P1[j].temp;

						D_lind = 0.1*tmp_hij*tmp_hij;
						ww=twij/w_dx_i;

						// int_t flag = ((ptypei==ptypej)||(ptypej<=0));
						int_t flag = 1;
						// int_t flag = (ptypei==ptypej);
						tmpx+=-D_lind*mj/rhoj*(1.0+0.2*(ww*ww*ww*ww))*tdwx*flag;
						tmpy+=-D_lind*mj/rhoj*(1.0+0.2*(ww*ww*ww*ww))*tdwy*flag;
						conc+=mj/rhoj*(1.0+0.2*(ww*ww*ww*ww))*twij*flag;

						tmprhox+=(rhoj-rhoi)*mj/rhoj*tdwx*flag;
						tmprhoy+=(rhoj-rhoi)*mj/rhoj*tdwy*flag;
						tmppresx+=(presj-presi)*mj/rhoj*tdwx*flag;
						tmppresy+=(presj-presi)*mj/rhoj*tdwy*flag;
						tmpuxx+=(uxj-uxi)*mj/rhoj*tdwx*flag;
						tmpuxy+=(uxj-uxi)*mj/rhoj*tdwy*flag;
						tmpuyx+=(uyj-uyi)*mj/rhoj*tdwx*flag;
						tmpuyy+=(uyj-uyi)*mj/rhoj*tdwy*flag;
						tmpmx+=(mj-mi)*mj/rhoj*tdwx*flag;
						tmpmy+=(mj-mi)*mj/rhoj*tdwy*flag;
						tmphx+=(Hj-Hi)*mj/rhoj*tdwx*flag;
						tmphy+=(Hj-Hi)*mj/rhoj*tdwy*flag;
						tmptx+=(Tj-Ti)*mj/rhoj*tdwx*flag;
						tmpty+=(Tj-Ti)*mj/rhoj*tdwy*flag;

						flt+=mj/rhoj*twij*(ptypej>0);
					}
				}
				// }
			}
		}
		}
	}

//Khayyer Optimized Particle Shifting

	Real tmpxx, tmpyy;

	Real nx = P3[i].nx;
	Real ny = P3[i].ny;
	Real lambda = P3[i].lambda;

	// if(lambda<0.85){
	// // 	tmpx=0.0;
	// // 	tmpy=0.0;
	if(lambda<0.6){
		tmpxx=0.0;
		tmpyy=0.0;
	}else if(lambda>=0.6&&lambda<0.85){
		tmpxx = tmpx*(1.0-nx*nx)-tmpy*nx*ny;
		tmpyy = tmpy*(1.0-ny*ny)-tmpx*nx*ny;
	}else{
		tmpxx=tmpx;
		tmpyy=tmpy;
	}

	P1[i].x=xi+tmpxx;
	P1[i].y=yi+tmpyy;
	P1[i].shiftx = tmpxx;
	P1[i].shifty = tmpyy;
	// P1[i].pres=presi+(tmppresx*tmpxx+tmppresy*tmpyy);
	// if(k_solver_type==Wcsph){
	// 	double B = k_rho0_eos*k_soundspeed*k_soundspeed/k_gamma;
	// 	double K = P1[i].pres/B + 1.0;
	// 	P1[i].rho = P2[i].rho_ref*pow(K, 1.0/k_gamma);
	// }
	// P1[i].ux=uxi+(tmpuxx*tmpxx+tmpuxy*tmpyy);
	// P1[i].uy=uyi+(tmpuyx*tmpxx+tmpuyy*tmpyy);
	P1[i].enthalpy=Hi+(tmphx*tmpxx+tmphy*tmpyy);
	P1[i].temp=Ti+(tmptx*tmpxx+tmpty*tmpyy);

}
////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_particle_shifting_lind2D2(int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3,Real tdt)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part3) return;
	if(P1[i].i_type==3) return;
	if(P1[i].p_type==9) return;
	if(P1[i].p_type<=0) return;
	if(P1[i].buffer_type>0) return;
	if((P1[i].elix<1e-10)&(P1[i].eliy<1e-10)) return;
	if(P1[i].p_type>=1000)	return;		// Immersed Boundary Method
	if(P1[i].p_type!=2)	return;		// Immersed Boundary Method
	if(P3[i].lbl_surf==3)	return;

	int_t ptypei;
	int_t icell,jcell;
	Real xi,yi,uxi,uyi,rhoi,mi,Hi,Ti,presi;
	Real w_dx_i,dr_square,hi;
	Real tmpx,tmpy,tmprhox,tmprhoy,tmppresx,tmppresy;
	Real tmpuxx, tmpuxy, tmpuyx, tmpuyy;
	Real tmpmx,tmpmy,tmphx,tmphy,tmptx,tmpty;
	Real flt;
	Real conc;
	Real nx, ny;

	Real search_range,tmp_A;

	hi=P1[i].h;

	ptypei=P1[i].p_type;

	xi=P1[i].x;
	yi=P1[i].y;
	uxi=P1[i].ux;
	uyi=P1[i].uy;
	rhoi=P1[i].rho;
	presi=P1[i].pres;
	mi=P1[i].m;
	Hi=P1[i].enthalpy;
	Ti=P1[i].temp;
	nx=P3[i].nx_s;
	ny=P3[i].ny_s;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;

	tmpx=tmpy=0.0;
	tmprhox=tmprhoy=0.0;
	tmppresx=tmppresy=0.0;
	tmpuxx=tmpuxy=tmpuyx=tmpuyy=0.0;
	tmpmx=tmpmy=tmphx=tmphy=tmptx=tmpty=0.0;
	conc=0.0;
	flt=0.0;

	for(int_t y=-P1[i].ncell;y<=P1[i].ncell;y++){
		for(int_t x=-P1[i].ncell;x<=P1[i].ncell;x++){
			int_t k=(icell+x)+k_NI*(jcell+y);
			if(k<0||k>=k_num_cells-1)	continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					Real xj,yj,tdist,elixj,eliyj,hj;
					Real uxj, uyj;
					int_t ptypej;

					xj=P1[j].x;
					yj=P1[j].y;
					hj=P1[j].h;

					Real hshift = hi;

					uxj=P1[j].ux;
					uyj=P1[j].uy;
					elixj=P1[j].elix;
					eliyj=P1[j].eliy;
					ptypej=P1[j].p_type;
					search_range=k_search_kappa*hshift;	// search range

						if(ptypej<1000){
							Real tmp_hij;
							if(k_multi_type==3||k_multi_type==4)	tmp_hij=(P1[i].h+P1[j].h)/2;
							else	tmp_hij=P1[i].h;
							Real tmp_Aij=calc_tmpA(tmp_hij);
		
							search_range=k_search_kappa*tmp_hij;
					tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+1E-20);
					if(tdist<search_range){
						Real tdwx,tdwy,tdwij,twij,hj,mj,rhoj,presj,ww,volj,Hj,Tj;
						Real D_lind;

						tdwij=calc_kernel_dwij(tmp_Aij,tmp_hij,tdist);
						twij=calc_kernel_wij(tmp_Aij,tmp_hij,tdist);
						w_dx_i=P1[i].w_dx;

						tdwx=tdwij*(xi-xj)/tdist;
						tdwy=tdwij*(yi-yj)/tdist;
						
						// apply_gradient_correction_2D(P3[i].Cm,twij,tdwx,tdwy,&tdwx,&tdwy);

						mj=P1[j].m;
						hj=P1[j].h;
						rhoj=P1[j].rho;
						presj=P1[j].pres;

						volj=P1[j].vol;
						Hj=P1[j].enthalpy;
						Tj=P1[j].temp;

						D_lind = 0.02*tmp_hij*tmp_hij;
						ww=twij/w_dx_i;

						int_t flag = ((ptypei==ptypej)||(ptypej<=0));
						tmpx+=-D_lind*mj/rhoj*(1.0+0.2*(ww*ww*ww*ww))*tdwx;
						tmpy+=-D_lind*mj/rhoj*(1.0+0.2*(ww*ww*ww*ww))*tdwy;
						conc+=mj/rhoj*(1.0+0.2*(ww*ww*ww*ww))*twij;

						tmprhox+=(rhoj-rhoi)*mj/rhoj*tdwx*flag;
						tmprhoy+=(rhoj-rhoi)*mj/rhoj*tdwy*flag;
						tmppresx+=(presj-presi)*mj/rhoj*tdwx*flag;
						tmppresy+=(presj-presi)*mj/rhoj*tdwy*flag;
						tmpuxx+=(uxj-uxi)*mj/rhoj*tdwx*flag;
						tmpuxy+=(uxj-uxi)*mj/rhoj*tdwy*flag;
						tmpuyx+=(uyj-uyi)*mj/rhoj*tdwx*flag;
						tmpuyy+=(uyj-uyi)*mj/rhoj*tdwy*flag;
						tmpmx+=(mj-mi)*mj/rhoj*tdwx*flag;
						tmpmy+=(mj-mi)*mj/rhoj*tdwy*flag;
						tmphx+=(Hj-Hi)*mj/rhoj*tdwx*flag;
						tmphy+=(Hj-Hi)*mj/rhoj*tdwy*flag;
						tmptx+=(Tj-Ti)*mj/rhoj*tdwx*flag;
						tmpty+=(Tj-Ti)*mj/rhoj*tdwy*flag;

						flt+=mj/rhoj*twij*(ptypej>0);
					}
				}
				// }
			}
		}
		}
	}

	P1[i].x=xi+tmpx;
	P1[i].y=yi+tmpy;
	P1[i].shiftx = tmpx;
	P1[i].shifty = tmpy;
	// P1[i].pres=presi+(tmppresx*tmpx+tmppresy*tmpy);
	// if(k_solver_type==Wcsph){
	// 	double B = k_rho0_eos*k_soundspeed*k_soundspeed/k_gamma;
	// 	double K = P1[i].pres/B + 1.0;
	// 	P1[i].rho = P2[i].rho_ref*pow(K, 1.0/k_gamma);
	// }
	// P1[i].ux=uxi+(tmpuxx*tmpx+tmpuxy*tmpy);
	// P1[i].uy=uyi+(tmpuyx*tmpx+tmpuyy*tmpy);
	P1[i].enthalpy=Hi+(tmphx*tmpx+tmphy*tmpy);
	P1[i].temp=Ti+(tmptx*tmpx+tmpty*tmpy);

}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_particle_shifting_oger2D(int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3,Real tdt,Real*u_mag)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type==3) return;
	if(P1[i].p_type==9) return;
	if(P1[i].p_type==0) return;
	if(P1[i].buffer_type>0) return;
	if((P1[i].elix<1e-10)&(P1[i].eliy<1e-10)) return;
	if(P1[i].p_type>=1000)	return;		// Immersed Boundary Method
	if(k_solver_type==Icsph&&P1[i].p_type==1)	return;		// Immersed Boundary Method

	int_t ptypei;
	int_t icell,jcell;
	Real xi,yi,uxi,uyi,rhoi,mi;
	Real w_dx_i,dr_square,hi;
	Real tmpx,tmpy,tmprhox,tmprhoy;
	Real tmpuxx, tmpuxy, tmpuyx, tmpuyy;
	Real tmpmx,tmpmy;
	Real num;
	Real conc;
	Real concx, concy, concs, concn;

	Real search_range,tmp_A;

	hi=P1[i].h;
	tmp_A=calc_tmpA(hi);
	search_range=k_search_kappa*hi;	// search range

	ptypei=P1[i].p_type;

	xi=P1[i].x;
	yi=P1[i].y;
	uxi=P1[i].ux;
	uyi=P1[i].uy;
	w_dx_i=P1[i].w_dx;
	rhoi=P1[i].rho;
	mi=P1[i].m;

	Real D_coeff,D_lind,D_skillen;

	Real umag = sqrt(uxi*uxi+uyi*uyi);
	// Real umag = u_mag[0];

	if(P1[i].p_type==2&&k_solver_type==Isph)	umag = 0.2*hi/tdt;
	// if(P1[i].p_type==2&&k_solver_type==Isph)	umag = u_mag[0];
	// D_lind = hi*sqrt(2.0*Gravitational_CONST*0.6);
	// D_lind = k_kappa*hi*umag;
	D_lind = k_kappa*hi*umag;
	// D_skillen = 4*hi*umag*tdt;
	// D_lind = 4*hi*1.0*tdt;

	D_coeff = max(D_lind,D_skillen);

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;

	tmpx=tmpy=0.0;
	tmprhox=tmprhoy=0.0;
	tmpuxx=tmpuxy=tmpuyx=tmpuyy=0.0;
	tmpmx=tmpmy=0.0;
	conc=0.0;
	for(int_t y=-P1[i].ncell;y<=P1[i].ncell;y++){
		for(int_t x=-P1[i].ncell;x<=P1[i].ncell;x++){
			int_t k=(icell+x)+k_NI*(jcell+y);
			if(k<0||k>=k_num_cells-1)	continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					Real xj,yj,tdist,elixj,eliyj;
					Real uxj, uyj;
					int_t ptypej;

					xj=P1[j].x;
					yj=P1[j].y;

					uxj=P1[j].ux;
					uyj=P1[j].uy;
					elixj=P1[j].elix;
					eliyj=P1[j].eliy;
					ptypej=P1[j].p_type;

					// if((elixj>1e-10)||(eliyj>1e-10)||(ptypej!=1)){
						if(P1[j].p_type<1000){

					tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+1E-20);
					Real tmp_hij;
					if(k_multi_type==3||k_multi_type==4)	tmp_hij=(P1[i].h+P1[j].h)/2;
					else	tmp_hij=P1[i].h;
					Real tmp_Aij=calc_tmpA(tmp_hij);

					search_range=k_search_kappa*tmp_hij;
					if(tdist<search_range){
						Real tdwx,tdwy,tdwij,twij,hj,mj,rhoj,ww,volj;
						tdwij=calc_kernel_dwij(tmp_Aij,tmp_hij,tdist);
						twij=calc_kernel_wij(tmp_Aij,tmp_hij,tdist);

						tdwx=tdwij*(xi-xj)/tdist;
						tdwy=tdwij*(yi-yj)/tdist;
						
						// apply_gradient_correction_2D(P3[i].Cm,twij,tdwx,tdwy,&tdwx,&tdwy);

						mj=P1[j].m;
						hj=P1[j].h;
						rhoj=P1[j].rho;
						volj=P1[j].vol;

						ww=twij/w_dx_i;
						tmpx+=-D_lind*mj/rhoj*(1.0+0.2*(ww*ww*ww*ww))*tdwx;
						tmpy+=-D_lind*mj/rhoj*(1.0+0.2*(ww*ww*ww*ww))*tdwy;
						conc+=mj/rhoj*(1.0+0.2*(ww*ww*ww*ww))*twij;
					}
				}
			}
		}
		}
	}

	// int flag = (!(ptypei==1&&P3[i].lambda<=0.9));
	// // int flag = 1;

	// P1[i].dux = tmpx*flag;
	// P1[i].duy = tmpy*flag;

	// Real tmpu = sqrt(P1[i].dux*P1[i].dux+P1[i].duy*P1[i].duy);
	// if(tmpu>=umag/2.0){
	// 	P1[i].dux = umag/2.0*(tmpx/tmpu);
	// 	P1[i].duy = umag/2.0*(tmpy/tmpu);
	// }

	// // tmpx = P1[i].dux;
	// // tmpy = P1[i].duy;

	Real tmpxx, tmpyy;

	Real nx = P3[i].nx_s;
	Real ny = P3[i].ny_s;
	Real lambda = P3[i].lambda;

	if(lambda<=0.6){
		tmpxx=0.0;
		tmpyy=0.0;
	}else if(lambda>0.6&&lambda<0.95){
		tmpxx = (lambda-0.6)/0.35*(tmpx*(1.0-nx*nx)-tmpy*nx*ny);
		tmpyy = (lambda-0.6)/0.35*(tmpy*(1.0-ny*ny)-tmpx*nx*ny);
	}else{
		tmpxx=tmpx;
		tmpyy=tmpy;
	}

	P1[i].dux = tmpxx;
	P1[i].duy = tmpyy;

	Real tmpu = sqrt(P1[i].dux*P1[i].dux+P1[i].duy*P1[i].duy);
	if(tmpu>=umag/2.0){
		P1[i].dux = umag/2.0*(tmpxx/tmpu);
		P1[i].duy = umag/2.0*(tmpyy/tmpu);
	}


	P1[i].shiftx = P1[i].dux;
	P1[i].shifty = P1[i].duy;
	P1[i].concentration = conc;
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_particle_shifting_oger2D_ISPH(int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3,Real tdt,Real*u_mag)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type==3) return;
	if(P1[i].p_type==9) return;
	if(P1[i].p_type==0) return;
	if(P1[i].buffer_type>0) return;
	if((P1[i].elix<1e-10)&(P1[i].eliy<1e-10)) return;
	if(P1[i].p_type>=1000)	return;		// Immersed Boundary Method
	if(k_solver_type==Icsph&&P1[i].p_type==2)	return;		// Immersed Boundary Method

	int_t ptypei;
	int_t icell,jcell;
	Real xi,yi,uxi,uyi,rhoi,mi;
	Real w_dx_i,dr_square,hi;
	Real tmpx,tmpy,tmprhox,tmprhoy;
	Real tmpuxx, tmpuxy, tmpuyx, tmpuyy;
	Real tmpmx,tmpmy;
	Real num;
	Real conc;
	Real concx, concy, concs, concn;

	Real search_range,tmp_A;

	hi=P1[i].h;
	tmp_A=calc_tmpA(hi);
	search_range=k_search_kappa*hi;	// search range

	ptypei=P1[i].p_type;

	xi=P1[i].x;
	yi=P1[i].y;
	uxi=P1[i].ux;
	uyi=P1[i].uy;
	w_dx_i=P1[i].w_dx;
	rhoi=P1[i].rho;
	mi=P1[i].m;

	Real D_coeff,D_lind,D_skillen;

	Real umag = sqrt(uxi*uxi+uyi*uyi);
	// D_lind = hi*sqrt(2.0*Gravitational_CONST*0.6);
	// D_lind = k_kappa*hi*umag;
	D_lind = k_kappa*hi*umag;
	// D_skillen = 4*hi*umag*tdt;
	// D_lind = 4*hi*1.0*tdt;

	D_coeff = max(D_lind,D_skillen);

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;

	tmpx=tmpy=0.0;
	tmprhox=tmprhoy=0.0;
	tmpuxx=tmpuxy=tmpuyx=tmpuyy=0.0;
	tmpmx=tmpmy=0.0;
	conc=0.0;
	for(int_t y=-P1[i].ncell;y<=P1[i].ncell;y++){
		for(int_t x=-P1[i].ncell;x<=P1[i].ncell;x++){
			int_t k=(icell+x)+k_NI*(jcell+y);
			if(k<0||k>=k_num_cells-1)	continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					Real xj,yj,tdist,elixj,eliyj;
					Real uxj, uyj;
					int_t ptypej;

					xj=P1[j].x;
					yj=P1[j].y;

					uxj=P1[j].ux;
					uyj=P1[j].uy;
					elixj=P1[j].elix;
					eliyj=P1[j].eliy;
					ptypej=P1[j].p_type;

					// if((elixj>1e-10)||(eliyj>1e-10)||(ptypej!=1)){
						if(P1[j].p_type<1000){

					tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+1E-20);
					Real tmp_hij;
					if(k_multi_type==3||k_multi_type==4)	tmp_hij=(P1[i].h+P1[j].h)/2;
					else	tmp_hij=P1[i].h;
					Real tmp_Aij=calc_tmpA(tmp_hij);

					search_range=k_search_kappa*tmp_hij;
					if(tdist<search_range){
						Real tdwx,tdwy,tdwij,twij,hj,mj,rhoj,ww,volj;
						tdwij=calc_kernel_dwij(tmp_Aij,tmp_hij,tdist);
						twij=calc_kernel_wij(tmp_Aij,tmp_hij,tdist);

						tdwx=tdwij*(xi-xj)/tdist;
						tdwy=tdwij*(yi-yj)/tdist;
						
						// apply_gradient_correction_2D(P3[i].Cm,twij,tdwx,tdwy,&tdwx,&tdwy);

						mj=P1[j].m;
						hj=P1[j].h;
						rhoj=P1[j].rho;
						volj=P1[j].vol;

						ww=twij/w_dx_i;
						tmpx+=-D_lind*mj/rhoj*(1.0+0.2*(ww*ww*ww*ww))*tdwx;
						tmpy+=-D_lind*mj/rhoj*(1.0+0.2*(ww*ww*ww*ww))*tdwy;
						conc+=mj/rhoj*(1.0+0.2*(ww*ww*ww*ww))*twij;
					}
				}
			}
		}
		}
	}

	int flag = (!(ptypei==1&&P3[i].lambda<=0.8));
	// int flag = 1;

	P1[i].dux = tmpx*flag;
	P1[i].duy = tmpy*flag;

	Real tmpu = sqrt(P1[i].dux*P1[i].dux+P1[i].duy*P1[i].duy);
	if(tmpu>=umag/2.0){
		P1[i].dux = umag/2.0*(tmpx/tmpu);
		P1[i].duy = umag/2.0*(tmpy/tmpu);
	}

	// tmpx = P1[i].dux;
	// tmpy = P1[i].duy;

	// Real tmpxx, tmpyy;

	// Real nx = P3[i].nx_s;
	// Real ny = P3[i].ny_s;
	// Real lambda = P3[i].lambda;

	// if(lambda<0.6&&P1[i].p_type==1){
	// 	Real tmp_x = tmpx;
	// 	Real tmp_y = tmpy;

	// 	Real tmpx_t = tmpx*(1.0-nx*nx)-tmpy*nx*ny;	//parallel
	// 	Real tmpy_t = tmpy*(1.0-ny*ny)-tmpx*nx*ny;	//parallel
	// 	Real tmpx_n = tmp_x-tmpx_t;
	// 	Real tmpy_n = tmp_y-tmpy_t;

	// 	tmpxx = tmpx_t+0.1*tmpx_n;	//parallel
	// 	tmpyy = tmpy_t+0.1*tmpy_n;	//parallel
	// }else{
	// 	tmpxx=tmpx;
	// 	tmpyy=tmpy;
	// }

	// P1[i].dux = tmpxx;
	// P1[i].duy = tmpyy;


	P1[i].shiftx = P1[i].dux;
	P1[i].shifty = P1[i].duy;
	P1[i].concentration = conc;
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_particle_shifting_lind3D(int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3, Real tdt)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part3) return;
	if(P1[i].i_type==3) return;

	int_t ptypei;
	int_t icell,jcell,kcell;
	Real xi,yi,zi;
	Real w_dx_i,hi,dr_square;
	Real search_range,tmp_A;
	Real tmpx,tmpy,tmpz;

	hi=P1[i].h;
	tmp_A=calc_tmpA(hi);
	search_range=k_search_kappa*hi;	// search range

	ptypei=P1[i].p_type;

	xi=P2[i].x0;
	yi=P2[i].y0;
	zi=P2[i].z0;
	w_dx_i=P1[i].w_dx;

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
	for(int_t z=-P1[i].ncell;z<=P1[i].ncell;z++){
		for(int_t y=-P1[i].ncell;y<=P1[i].ncell;y++){
			for(int_t x=-P1[i].ncell;x<=P1[i].ncell;x++){
				int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
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
							Real tdwx,tdwy,tdwz,tdwij,twij,mj,hj,rhoj,ww;
							tdwij=calc_kernel_dwij(tmp_A,hi,tdist);
							twij=calc_kernel_wij(tmp_A,hi,tdist);
							if(k_kgc_solve==1){
								// tdwx=((P3[i].inv_cm_xx*tdwij*(xi-xj)/tdist)+(P3[i].inv_cm_xy*tdwij*(yi-yj)/tdist)+(P3[i].inv_cm_zx*tdwij*(zi-zj)/tdist));
								// tdwy=((P3[i].inv_cm_xy*tdwij*(xi-xj)/tdist)+(P3[i].inv_cm_yy*tdwij*(yi-yj)/tdist)+(P3[i].inv_cm_yz*tdwij*(zi-zj)/tdist));
								// tdwz=((P3[i].inv_cm_zx*tdwij*(xi-xj)/tdist)+(P3[i].inv_cm_yz*tdwij*(yi-yj)/tdist)+(P3[i].inv_cm_zz*tdwij*(zi-zj)/tdist));
							}else{
								tdwx=tdwij*(xi-xj)/tdist;
								tdwy=tdwij*(yi-yj)/tdist;
								tdwz=tdwij*(zi-zj)/tdist;
							}

							mj=P1[j].m;
							rhoj=P1[j].rho;
							hj=P1[j].h;

							ww=twij/w_dx_i;
							tmpx+=-0.02*hj*hj*mj/rhoj*(0.2*(ww*ww*ww*ww))*tdwx;
							tmpy+=-0.02*hj*hj*mj/rhoj*(0.2*(ww*ww*ww*ww))*tdwy;
							tmpz+=-0.02*hj*hj*mj/rhoj*(0.2*(ww*ww*ww*ww))*tdwz;
						}
					}
				}
			}
		}
	}

	dr_square=tmpx*tmpx+tmpy*tmpy+tmpz*tmpz;

	// if ((P3[i].lbl_surf<0.5)&(dr_square<0.01*hi*hi)){		// interior
	// 	P2[i].x0=xi+tmpx*(ptypei>0);
	// 	P2[i].y0=yi+tmpy*(ptypei>0);
	// 	P2[i].z0=zi+tmpz*(ptypei>0);
	// }

	if ((dr_square<0.01*hi*hi)){		// interior
		P2[i].x0=xi+tmpx*(ptypei>0);
		P2[i].y0=yi+tmpy*(ptypei>0);
		P2[i].z0=zi+tmpz*(ptypei>0);
	}
}
////////////////////////////////////////////////////////////////////////
// Gaussian Kernel function
__global__ void KERNEL_clc_gaussian_w_dx(part1*P1)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part3) return;
	if(P1[i].i_type==3) return;

	Real tmp_R,tmp_A;
	Real hi = P1[i].h;
	//dx/h=(2/3*h)/h
	tmp_R=2.0/3.0;

	//if(k_dim==1) tmp_A=1.0/(pow(PI,0.5)*P1[i].h);
	if(k_dim==2) tmp_A=1.0/(PI*pow(hi,2));
	if(k_dim==3) tmp_A=1.0/(pow(PI,1.5)*pow(hi,3));
	P1[i].w_dx=tmp_A*exp(-pow(tmp_R,2));
}
////////////////////////////////////////////////////////////////////////
// Quintic kernel
__global__ void KERNEL_clc_quintic_w_dx(part1*P1)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part3) return;
	if(P1[i].i_type==3) return;

	Real tmp_R,tmp_A;
	Real hi=P1[i].h;

	hi=P1[i].h;
	// tmp_R=P1[i].h/1.5/(hi);
	tmp_R=1.0;
	//if(k_dim==1) tmp_A=1.0;
	if(k_dim==2) tmp_A=7.0/(478.0*PI*pow(hi,2));
	if(k_dim==3) tmp_A=3.0/(359.0*PI*pow(hi,3));

	Real tmpwdx;
	tmpwdx=pow(3.0-tmp_R,5);
	tmpwdx+=-6.0*pow(2.0-tmp_R,5);
	tmpwdx+=15.0*pow(1.0-tmp_R,5);
	tmpwdx*=tmp_A;
	P1[i].w_dx=tmpwdx;
	//P1[i].w_dx=tmp_A*(pow(3.0-tmp_R,5)-6.0*pow(2.0-tmp_R,5)+15.0*pow(1.0-tmp_R,5));
}
////////////////////////////////////////////////////////////////////////
// Quartic kernel
__global__ void KERNEL_clc_quartic_w_dx(part1*P1)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part3) return;
	if(P1[i].i_type==3) return;

	Real tmp_R,tmp_A;
	Real hi=P1[i].h;

	//dx/h=(2/3*h)/h
	tmp_R=2.0/3.0;

	//if(k_dim==1) tmp_A=1.0/P1[i].h;
	if(k_dim==2) tmp_A=15.0/(7.0*PI*pow(hi,2));
	if(k_dim==3) tmp_A=315.0/(208.0*PI*pow(hi,3));

	Real tmpwdx=0.0;
	if(tmp_R<2){
		tmpwdx=2.0/3.0-9.0/8.0*pow(tmp_R,2);
		tmpwdx+=19.0/24.0*pow(tmp_R,3);
		tmpwdx+=-5.0/32.0*pow(tmp_R,4);
		tmpwdx*=tmp_A;
	}
	P1[i].w_dx=tmpwdx;
	//P1[i].w_dx=(tmp_R<2)*tmp_A*(2.0/3.0-9.0/8.0*pow(tmp_R,2)+19.0/24.0*pow(tmp_R,3)-5.0/32.0*pow(tmp_R,4));
}
////////////////////////////////////////////////////////////////////////
// Wendland2 kernel
__global__ void KERNEL_clc_wendland2_w_dx(part1*P1)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part3) return;
	if(P1[i].i_type==3) return;

	Real tmp_R,hi,tmp_C,tmpwdx;

	//dx/2h=(2/3*h)/h*0.5
	hi=P1[i].h;
	tmp_R=P1[i].h/1.5/(hi*k_kappa);

	// equation of Wendland 2 kernel function

	// if(k_dim==1){
	// 	tmp_C=1.25/(2*hi);// 5./(4*(2h))
	// 	P1[i].w_dx=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1+3*tmp_R);
	// }

	if(k_dim==2){
		tmp_C=2.228169203286535/(k_kappa*k_kappa*hi*hi);				// 7.0/(pi*(2h)^2)
		tmpwdx=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R);
		tmpwdx*=(1-tmp_R)*(1-tmp_R)*(1+4*tmp_R);
		P1[i].w_dx=tmpwdx;
		//P1[i].w_dx=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1+4*tmp_R);
	}
	if(k_dim==3){
		tmp_C=3.342253804929802/(k_kappa*k_kappa*k_kappa*hi*hi*hi);	// 21.0/(2*pi*(2h)^3)
		tmpwdx=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R);
		tmpwdx*=(1-tmp_R)*(1-tmp_R)*(1+4*tmp_R);
		P1[i].w_dx=tmpwdx;
		//P1[i].w_dx=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1+4*tmp_R);
	}
}
////////////////////////////////////////////////////////////////////////
// Wendland4 kernel
__global__ void KERNEL_clc_wendland4_w_dx(part1*P1)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part3) return;
	if(P1[i].i_type==3) return;

	Real tmp_R,hi,tmp_C,tmpwdx;

	//dx/2h=(2/3*h)/h*0.5
	hi=P1[i].h;
	tmp_R=P1[i].h/1.5/(hi*k_kappa);

	// equation of Wendland 4 kernel function

	// if(k_dim==1){
	// 	tmp_C=1.5/(2*hi);// 3./(2*(2h))
	// 	P1[i].w_dx=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1+5*tmp_R+8*tmp_R*tmp_R);
	// }

	if(k_dim==2){
		tmp_C=2.864788975654116/(k_kappa*k_kappa*hi*hi);					// 9./(pi*(2hi)^2)
		tmpwdx=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R);
		tmpwdx*=(1-tmp_R)*(1-tmp_R)*(1-tmp_R);
		tmpwdx*=(1-tmp_R)*(1+6*tmp_R+11.666666666666666*tmp_R*tmp_R);
		P1[i].w_dx=tmpwdx;
		//P1[i].w_dx=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1+6*tmp_R+11.666666666666666*tmp_R*tmp_R);
	}
	if(k_dim==3){
		tmp_C=4.923856051905513/(k_kappa*k_kappa*k_kappa*hi*hi*hi);		// 495./(32*pi*(2hi)^3)
		tmpwdx=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R);
		tmpwdx*=(1-tmp_R)*(1-tmp_R)*(1-tmp_R);
		tmpwdx*=(1-tmp_R)*(1+6*tmp_R+11.666666666666666*tmp_R*tmp_R);
		P1[i].w_dx=tmpwdx;
		//P1[i].w_dx=(tmp_R<1)*tmp_C* (1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1+6*tmp_R+11.666666666666666*tmp_R*tmp_R);
	}
}
////////////////////////////////////////////////////////////////////////
// Wendland6 kernel
__global__ void KERNEL_clc_wendland6_w_dx(part1*P1)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part3) return;
	if(P1[i].i_type==3) return;

	Real tmp_R,hi,tmp_C,tmpwdx;

	hi=P1[i].h;
	tmp_R=P1[i].h/1.5/(hi*k_kappa);
	// tmp_R=1.0/3.0;

	if(k_dim==2){
		tmp_C=3.546881588905096/(k_kappa*k_kappa*hi*hi);					// 9./(pi*(2hi)^2)
		tmpwdx=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R)*(1-tmp_R);
		tmpwdx*=(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R);
		tmpwdx*=(1-tmp_R)*(1+8*tmp_R+25*tmp_R*tmp_R+32*tmp_R*tmp_R*tmp_R);
		P1[i].w_dx=tmpwdx;
	}
	if(k_dim==3){
		tmp_C=6.788953041263660/(k_kappa*k_kappa*k_kappa*hi*hi*hi);		// 495./(32*pi*(2hi)^3)
		tmpwdx=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R)*(1-tmp_R);
		tmpwdx*=(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R);
		tmpwdx*=(1-tmp_R)*(1+8*tmp_R+25*tmp_R*tmp_R+32*tmp_R*tmp_R*tmp_R);
		P1[i].w_dx=tmpwdx;
	}
}
////////////////////////////////////////////////////////////////////////
void calculate_w_dx(part1*P1)
{
	dim3 b,t;
	t.x=128;
	b.x=(num_part3-1)/t.x+1;

	// Calculate kernel value for initial spacing
	if(kernel_type==Gaussian)	 KERNEL_clc_gaussian_w_dx<<<b,t>>>(P1);
	if(kernel_type==Quintic)	 KERNEL_clc_quintic_w_dx<<<b,t>>>(P1);
	if(kernel_type==Quartic)	 KERNEL_clc_quartic_w_dx<<<b,t>>>(P1);
	if(kernel_type==Wendland2) KERNEL_clc_wendland2_w_dx<<<b,t>>>(P1);
	if(kernel_type==Wendland4) KERNEL_clc_wendland4_w_dx<<<b,t>>>(P1);
	if(kernel_type==Wendland6) KERNEL_clc_wendland6_w_dx<<<b,t>>>(P1);
	cudaDeviceSynchronize();
}

////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_normalvectorforpst(int_t*g_str,int_t*g_end,part1*P1, part2*P2, part3*P3, int_t tcount)
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
	Real li = P3[i].lambda;
	Real grad_lix, grad_liy;

	xi=P1[i].x;
    yi=P1[i].y;

	uxi=P1[i].ux;
	uyi=P1[i].uy;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}

	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;

	// 변수 초기화
	grad_lix=grad_liy=0.0;

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
					Real tmp_Aij=calc_tmpA(tmp_hij);

					search_range=k_search_kappa*tmp_hij;
					if(tdist<search_range){

						tmp_wij=calc_kernel_wij(tmp_Aij,tmp_hij,tdist);
						tmp_dwij=calc_kernel_dwij(tmp_Aij,tmp_hij,tdist);

						// dwij
						tdwx=tmp_dwij*(xi-xj)/tdist;
						tdwy=tmp_dwij*(yi-yj)/tdist;

							Real tdwxc=P3[i].Cm[0][0]*tdwx+P3[i].Cm[0][1]*tdwy;
							Real tdwyc=P3[i].Cm[1][0]*tdwx+P3[i].Cm[1][1]*tdwy;

							if(li>=0.7){
								grad_lix+=(lj-li)*tdwxc*mj/rhoj;
								grad_liy+=(lj-li)*tdwyc*mj/rhoj;
							}else{
								grad_lix+=(lj)*tdwxc*mj/rhoj;
								grad_liy+=(lj)*tdwyc*mj/rhoj;
							}

					}
				}
			}
				}
			}
		}
	}

	Real gradli = sqrt(grad_lix*grad_lix+grad_liy*grad_liy);
	if(gradli>0.1*li/tmp_h && P1[i].p_type==1){
		Real lambdamag = sqrt(grad_lix*grad_lix + grad_liy*grad_liy);
		P3[i].nx = -grad_lix/lambdamag;
		P3[i].ny = -grad_liy/lambdamag;
		P3[i].nmag = sqrt(P3[i].nx*P3[i].nx+P3[i].ny*P3[i].ny+P3[i].nz*P3[i].nz);
	}else{
		P3[i].nx = 0.0;
		P3[i].ny = 0.0;
		P3[i].nmag = 0.0;

	}

}

// Particle Shifting Algorithm: computes max velocity and applies shifting if enabled
void shifting(
    dim3& b, dim3 t,
    int_t* g_str, int_t* g_end,
    part1* dev_P1, part2* dev_SP2, part3* dev_P3
) {
    // Allocate device arrays for velocity max computation
    Real* max_umag = nullptr;
    Real* d_max_umag0 = nullptr;
    cudaMalloc((void**)&max_umag, sizeof(Real) * num_part3);
    cudaMalloc((void**)&d_max_umag0, sizeof(Real));
    cudaMemset(max_umag, 0, sizeof(Real) * num_part3);
    cudaMemset(d_max_umag0, 0, sizeof(Real));

    // Allocate CUB workspace for max reduction
    void* dev_max_storage = nullptr;
    size_t max_storage_bytes = 0;
    cub::DeviceReduce::Max(dev_max_storage, max_storage_bytes, max_umag, d_max_umag0, num_part3);
    cudaDeviceSynchronize();
    cudaMalloc(&dev_max_storage, max_storage_bytes);

    // Compute max velocity on device
    kernel_copy_max_velocity<<<b, t>>>(dev_P1, dev_SP2, dev_P3, max_umag);
    cudaDeviceSynchronize();

    cub::DeviceReduce::Max(dev_max_storage, max_storage_bytes, max_umag, d_max_umag0, num_part3);
    cudaDeviceSynchronize();

    // Particle Shifting Technique (PST)
    b.x = (num_part3 - 1) / t.x + 1;
    if (pst_solve == 1) {
        calculate_w_dx(dev_P1);
        cudaDeviceSynchronize();
        if (dim == 2)
            KERNEL_clc_particle_shifting_oger2D<<<b, t>>>(g_str, g_end, dev_P1, dev_SP2, dev_P3, dt, d_max_umag0);
        cudaDeviceSynchronize();
    }

    cudaFree(d_max_umag0);
    cudaFree(max_umag);
    cudaFree(dev_max_storage);
}
