////////////////////////////////////////////////////////////////////////
__device__ Real calc_tmpA(Real tH)
{
	Real tA=0.0;
	//
	if(k_kernel_type==Gaussian){
		if(k_dim==2) tA=1.0/(PI*pow(tH,2));
		else if(k_dim==3) tA=1.0/(pow(PI,1.5)*pow(tH,3));
	}else	if(k_kernel_type==Quintic){
		if(k_dim==2) tA=7.0/(478.0*PI*pow(tH,2));
		else if(k_dim==3) tA=3.0/(359.0*PI*pow(tH,3));
	}else	if(k_kernel_type==Quartic){
		if(k_dim==2) tA=15.0/(7.0*PI*pow(tH,2));
		else if(k_dim==3) tA=315.0/(208.0*PI*pow(tH,3));
	}else	if(k_kernel_type==Wendland2){
		if(k_dim==2) tA=2.228169203286535/(k_kappa*tH)/(k_kappa*tH);	// 7.0/(pi*(2h)^2)
		else if(k_dim==3) tA=3.342253804929802/(8.0*tH*tH*tH);	// 21.0/(2*pi*(2th)^3)
	}else	if(k_kernel_type==Wendland4){
		if(k_dim==2) tA=2.864788975654116/(k_kappa*tH)/(k_kappa*tH);	// 9.0/(pi*(2h)^2)
		else if(k_dim==3) tA=4.923856051905513/(8.0*tH*tH*tH);	// 495.0/(32*pi*(2h)^3)
	}else	if(k_kernel_type==Wendland6){
		if(k_dim==2) tA=3.546881588905096/(k_kappa*tH)/(k_kappa*tH);	// 78.0/(7*pi*(2h)^2)
		else if(k_dim==3) tA=6.788953041263660/(8.0*tH*tH*tH);	// 1365.0/(64*pi*(2h)^3)
	}
	return tA;
}
// ////////////////////////////////////////////////////////////////////////
__device__ Real calc_kernel_wij(Real tA,Real tH,Real rr){

	Real tR,wij;
	tR=wij=0.0;

	if(k_kernel_type==Gaussian){
		tR=rr/tH/k_kappa;
		wij=tA*exp(-pow(2.0*tR,2));
	}else	if(k_kernel_type==Quintic){
		tR=rr/tH/k_kappa;
		if(tR<0.5) wij=tA*(pow(3.0-2.0*tR,5)-6.0*pow(2.0-2.0*tR,5)+15.0*pow(1.0-2.0*tR,5));
		else if(0.5<=tR&&tR<1) wij=tA*(pow(3.0-2.0*tR,5)-6.0*pow(2.0-2.0*tR,5));
		else if(1<=tR&&tR<1.5) wij=tA*(pow(3.0-2.0*tR,5));
	}else	if(k_kernel_type==Quartic){
		tR=rr/tH/k_kappa;
		wij=(tR<1)*tA*(2.0/3.0-9.0/2.0*pow(tR,2)+19.0/3.0*pow(tR,3)-5.0/2.0*pow(tR,4));
	}else	if(k_kernel_type==Wendland2){
		tR=rr/tH/k_kappa;
		if(k_dim==2) wij=(tR<1)*tA*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1+4*tR);
		else if(k_dim==3) wij=(tR<1)*tA*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1+4*tR);
	}else	if(k_kernel_type==Wendland4){
		tR=rr/tH/k_kappa;
		if(k_dim==2) wij=(tR<1)*tA*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1+6*tR+11.666666666666666*tR*tR);
		else if(k_dim==3) wij=(tR<1)*tA*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1+6*tR+11.666666666666666*tR*tR);
	}else	if(k_kernel_type==Wendland6){
		tR=rr/tH/k_kappa;
		if(k_dim==2) wij=(tR<1)*tA*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1+8*tR+25*tR*tR+32*tR*tR*tR);
		else if(k_dim==3) wij=(tR<1)*tA*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1+8*tR+25*tR*tR+32*tR*tR*tR);
	}
	return wij;
}
////////////////////////////////////////////////////////////////////////
__device__ Real calc_kernel_dwij(Real tA,Real tH,Real rr){

	Real tR,dwij;
	tR=dwij=0.0;

	if(k_kernel_type==Gaussian){
		tR=rr/tH/k_kappa;
		dwij=(1.0/tH)*tA*(-2.0)*2.0*tR*exp(-pow(2.0*tR,2));
	}else	if(k_kernel_type==Quintic){
		tR=rr/tH/k_kappa;
		if(tR<0.5) dwij=(1.0/tH)*tA*(-5.0*pow(3.0-2.0*tR,4)+30.0*pow(2.0-2.0*tR,4)-75.0*pow(1.0-2.0*tR,4));
		else if(0.5<=tR&&tR<1.0) dwij=(1.0/tH)*tA*(-5.0*pow(3.0-2.0*tR,4)+30.0*pow(2.0-2.0*tR,4));
		else if(1.0<=tR&&tR<1.5) dwij=(1.0/tH)*tA*(-5.0*pow(3.0-2.0*tR,4));
	}else	if(k_kernel_type==Quartic){
		tR=rr/tH/k_kappa;
		dwij=(tR<1)*(1.0/tH)*tA*(-9.0/2.0*2*tR+19.0/3.0*3.0*pow(tR,2)-5.0/2.0*4.0*pow(tR,3));
	}else	if(k_kernel_type==Wendland2){
		tR=rr/tH/k_kappa;
		if(k_dim==2) dwij=(tR<1)*(1/(k_kappa*tH))*tA*(-20*tR*(1-tR)*(1-tR)*(1-tR));
		else if(k_dim==3) dwij=(tR<1)*(1/(k_kappa*tH))*tA*(-20*tR*(1-tR)*(1-tR)*(1-tR));
	}else	if(k_kernel_type==Wendland4){
		tR=rr/tH/k_kappa;
		if(k_dim==2) dwij=(tR<1)*(1/(k_kappa*tH))*tA*(-18.666666666666668*tR*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(5*tR+1));
		else if(k_dim==3) dwij=(tR<1)*(1/(k_kappa*tH))*tA*(-18.666666666666668*tR*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(5*tR+1));
	}else	if(k_kernel_type==Wendland6){
		tR=rr/tH/k_kappa;
		if(k_dim==2) dwij=(tR<1)*(1/(k_kappa*tH))*tA*(-22*tR*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(16*tR*tR+7*tR+1));
		else if(k_dim==3) dwij=(tR<1)*(1/(k_kappa*tH))*tA*(-22*tR*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(16*tR*tR+7*tR+1));
	}
	return dwij;
}
////////////////////////////////////////////////////////////////////////
__host__ __device__ void apply_gradient_correction_2D(Real Cm[][Correction_Matrix_Size],Real wij, Real dwx,Real dwy,Real* dwcx,Real* dwcy)
{
	// if(k_kgc_solve==KGC) {
		*dwcx=Cm[0][0]*dwx+Cm[0][1]*dwy;
		*dwcy=Cm[1][0]*dwx+Cm[1][1]*dwy;
	// }
	// else if(k_kgc_solve==FPM) {
	// 	*dwcx=Cm[0][0]*dwx+Cm[0][1]*dwy+Cm[0][2]*dwz;
	// 	*dwcy=Cm[1][0]*dwx+Cm[1][1]*dwy+Cm[1][2]*dwz;
	// 	*dwcz=Cm[2][0]*dwx+Cm[2][1]*dwy+Cm[2][2]*dwz;
	// }

}
////////////////////////////////////////////////////////////////////////
__host__ __device__ void apply_gradient_correction_3D(Real Cm[][Correction_Matrix_Size],Real wij,Real dwx,Real dwy,Real dwz,Real* dwcx,Real* dwcy,Real* dwcz)
{
	if(k_kgc_solve==KGC) {
		*dwcx=Cm[0][0]*dwx+Cm[0][1]*dwy+Cm[0][2]*dwz;
		*dwcy=Cm[1][0]*dwx+Cm[1][1]*dwy+Cm[1][2]*dwz;
		*dwcz=Cm[2][0]*dwx+Cm[2][1]*dwy+Cm[2][2]*dwz;
	}
	else if(k_kgc_solve==DFPM) {
		// *dwcx=Cm[0][0]*dwx+Cm[0][1]*dwy;
		// *dwcy=Cm[1][0]*dwx+Cm[1][1]*dwy;
	}
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_invA_MLS_2D(int_t*g_str,int_t*g_end,part1*P1,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part3) return;
	if(P1[i].i_type>i_type_crt) return;
	if(P1[i].p_type<1000)	return;		// Immersed Boundary Method

	int_t icell,jcell;
	Real xi,yi;
	Real search_range,tmp_h,tmp_A;;
	Real tmpxx,tmpyy,tmpzz,tmpxy,tmpyz,tmpzx;

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

	tmpxx=tmpyy=tmpzz=0;
	tmpxy=tmpyz=tmpzx=0;
	for(int_t y=-P1[i].ncell;y<=P1[i].ncell;y++){
		for(int_t x=-P1[i].ncell;x<=P1[i].ncell;x++){
			// int_t k=(icell+x)+k_NI*(jcell+y);
			int_t k=idx_cell(icell+x,jcell+y,0);

			if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))) continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					Real xj,yj,tdist;
					Real hj;
					int itype;
					int p_type_j, buffer_typej;
					hj=P1[i].h;

					xj=P1[j].x;
					yj=P1[j].y;
					itype=P1[j].i_type;
					buffer_typej=P1[j].buffer_type;

					// if((itype!=2)|((buffer_typej!=Inlet)&(buffer_typej!=Outlet))){
					if(itype!=4){
					if(P1[j].p_type<1000){

					tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj))+1e-20;
					if(tdist<search_range){
							Real mj,rhoj,txx,txy,tyy,tzx,tyz,tzz,rtd,mtd;
							Real twij=calc_kernel_wij(tmp_A,tmp_h,tdist);
							
							mj=P1[j].m;
							rhoj=P1[j].rho;

							mtd=twij;
							rtd=mj/(rhoj);

							txx=mtd;
							txx*=rtd;
							txy=mtd*(xi-xj)/hj;
							txy*=rtd;
							tyy=mtd*(xi-xj)*(xi-xj)/hj/hj;
							tyy*=rtd;
							tzx=mtd*(yi-yj)/hj;
							tzx*=rtd;
							tyz=mtd*(yi-yj)*(xi-xj/hj/hj);
							tyz*=rtd;
							tzz=mtd*(yi-yj)*(yi-yj)/hj/hj;
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
	}
	// save values to particle array
	Real tmpcmd;
	tmpcmd=tmpxx*(tmpyy*tmpzz-tmpyz*tmpyz);
	tmpcmd-=tmpxy*(tmpxy*tmpzz-tmpyz*tmpzx);
	tmpcmd+=tmpzx*(tmpxy*tmpyz-tmpyy*tmpzx);

	if(abs(tmpcmd)>Min_det){
		Real rtcmd=1.0/tmpcmd;
		P3[i].A[0][0]=(tmpyy*tmpzz-tmpyz*tmpyz)*rtcmd;
		P3[i].A[0][1]=(tmpzx*tmpyz-tmpxy*tmpzz)*rtcmd;
		P3[i].A[0][2]=(tmpxy*tmpyz-tmpzx*tmpyy)*rtcmd;
		P3[i].A[1][0]=(tmpzx*tmpyz-tmpxy*tmpzz)*rtcmd;
		P3[i].A[1][1]=(tmpxx*tmpzz-tmpzx*tmpzx)*rtcmd;
		P3[i].A[1][2]=(tmpzx*tmpxy-tmpxx*tmpyz)*rtcmd;
		P3[i].A[2][0]=(tmpxy*tmpyz-tmpzx*tmpyy)*rtcmd;
		P3[i].A[2][1]=(tmpzx*tmpxy-tmpxx*tmpyz)*rtcmd;
		P3[i].A[2][2]=(tmpxx*tmpyy-tmpxy*tmpxy)*rtcmd;
	}
	else{
		P3[i].A[0][0]=1;
		P3[i].A[0][1]=0;
		P3[i].A[0][2]=0;
		P3[i].A[1][0]=0;
		P3[i].A[1][1]=1;
		P3[i].A[1][2]=0;
		P3[i].A[2][0]=0;
		P3[i].A[2][1]=0;
		P3[i].A[2][2]=1;
	}

	// printf("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",tmpxx,tmpxy,tmpzx,tmpxy,tmpyy,tmpyz,tmpzx,tmpyz,tmpzz);
	// printf("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",P3[i].A[0][0],P3[i].A[1][0],P3[i].A[2][0],P3[i].A[0][1],P3[i].A[1][1],P3[i].A[2][1],P3[i].A[0][2],P3[i].A[1][2],P3[i].A[2][2]);

}
////////////////////////////////////////////////////////////////////////
__host__ __device__ void apply_MLS_filter_2D(Real A[][Correction_Matrix_Size],Real wij, Real *wij_mls, Real xi, Real xj, Real yi, Real yj, Real hj)
{
	// if(k_kgc_solve==KGC) {
		*wij_mls = (A[0][0]+A[1][0]*(xi-xj)/hj+A[2][0]*(yi-yj)/hj)*wij;
	// }
}
