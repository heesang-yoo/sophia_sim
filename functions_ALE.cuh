// ////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_set_ncell(part1*P1,int tcount)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part3) return;
	if(P1[i].i_type==3) return;

	Real h0=k_h_max;
	Real h=P1[i].h;

	// P1[i].ncell = ceil(ncell_init*(h/h0));
	P1[i].ncell = ceil(float(ncell_init/float(h0/h)));
	if(tcount==0)	P1[i].h0=P1[i].h;
	P1[i].h=P1[i].h0;
}
// ////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_set_alpha_Lagrangian(part1*P1)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part3) return;
	if(P1[i].i_type==3) return;

	P1[i].elix=1.0;
	P1[i].eliy=1.0;

}
// ////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_set_alpha(part1*P1)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part3) return;
	if(P1[i].i_type==3) return;
	if(P1[i].p_type==1000) return;

	
	Real xi = P1[i].x;
	Real yi = P1[i].y;

	// Real xcen = 0.08;
	// Real ycen = 0.09;
	// Real rad1x = 0.03;
	// Real rad2x = 0.045;
	// Real rad1y = 0.03;
	// Real rad2y = 0.05;

	
	// if(((abs(xi-xcen)>rad1x))&((abs(xi-xcen)<=rad2x)&(abs(yi-ycen)<=rad1y))){
	// // // if(((abs(xi-xcen)>rad1x)||(abs(yi-ycen)>rad1y))&((abs(xi-xcen)<=rad2x)&(abs(yi-ycen)<=rad2y))){
	// 	Real r1 = min(abs(xi-((xcen+rad1x))),abs(xi-((xcen-rad1x))));
		
	// 	Real radius = (rad2x-rad1x)/2.0;
	// 	Real norm = (exp(2.0*(1.0))-1.0)/(exp(2.0*(1.0))+1.0);
	// 	P1[i].elix=0.5+0.5*((exp(2.0*(1.0*(r1-radius)/(radius)))-1.0)/(exp(2.0*(1.0*(r1-radius)/(radius)))+1.0))/norm;
	// 	P1[i].eliy=0.5+0.5*((exp(2.0*(1.0*(r1-radius)/(radius)))-1.0)/(exp(2.0*(1.0*(r1-radius)/(radius)))+1.0))/norm;
	// }

	// if((abs(P1[i].x-xcen)>rad2x)||(abs(P1[i].y-ycen)>rad1y)){
	// // if((abs(P1[i].x-xcen)>rad2x)||(abs(P1[i].y-ycen)>rad2y)){
	// 	P1[i].elix=1.0;
	// 	P1[i].eliy=1.0;
	// }

	Real xcen = 0.0;
	Real ycen = 0.5;
	Real radius = 0.25;
	Real rad1x = 0.25;
	Real rad2x = 0.35;
	
	P1[i].elix=1.0;
	P1[i].eliy=1.0;

	if(((abs(xi-xcen)>rad1x-P1[i].h/h_coeff*0.5))&((abs(xi-xcen)<=rad2x+P1[i].h/h_coeff*0.5)&(abs(yi-ycen)<=rad1x+P1[i].h/h_coeff*0.5))){
			Real r1 = min(abs(xi-((xcen+rad1x)-P1[i].h/h_coeff*0.5)),abs(xi-((xcen-rad1x)+P1[i].h/h_coeff*0.5)));
			
			Real radius = (rad2x-rad1x)/2.0+P1[i].h/h_coeff*0.5;
			Real norm = (exp(2.0*(1.0))-1.0)/(exp(2.0*(1.0))+1.0);
			P1[i].elix=0.5+0.5*((exp(2.0*(1.0*(r1-radius)/(radius)))-1.0)/(exp(2.0*(1.0*(r1-radius)/(radius)))+1.0))/norm;
			P1[i].eliy=0.5+0.5*((exp(2.0*(1.0*(r1-radius)/(radius)))-1.0)/(exp(2.0*(1.0*(r1-radius)/(radius)))+1.0))/norm;
		}
	if((abs(P1[i].x-xcen)<=radius+1e-6)&&(abs(P1[i].y-ycen)<=radius+1e-6)){
		P1[i].elix=0.0;
		P1[i].eliy=0.0;
	}

	// if((P1[i].x-xcen)*(P1[i].x-xcen)+(P1[i].y-ycen)*(P1[i].y-ycen)<radius*radius){
	// 	P1[i].elix=0.0;
	// 	P1[i].eliy=0.0;
	// }
}
// ////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_smoothing_length2D(int_t*g_str,int_t*g_end,part1*P1){
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	  if(i>=k_num_part2) return;
	  if(P1[i].i_type>i_type_crt) return;
  
	  Real tmp_h=P1[i].h;
  
	  int_t icell,jcell;
	  Real xi,yi;
	  Real search_range,tmp_A,tmp_R,tmp_flt;
  
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
  
	tmp_R=0.0;
	tmp_flt=0.0;
	for(int_t y=-P1[i].ncell;y<=P1[i].ncell;y++){
		  for(int_t x=-P1[i].ncell;x<=P1[i].ncell;x++){
			  // int_t k=(icell+x)+k_NI*(jcell+y);
			  int_t k=idx_cell(icell+x,jcell+y,0);
			  if (k<0) continue;
			  if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))) continue;
			  if(g_str[k]!=cu_memset){
				  int_t fend=g_end[k];
				  for(int_t j=g_str[k];j<fend;j++){
					  Real xj,yj,tdist,tmp_hj,tmp_hij,tmp_Aij;
  
					  xj=P1[j].x;
					  yj=P1[j].y;
					  tmp_hj=P1[j].h;
					  tmp_hij=(tmp_h+tmp_hj)/2;
					  tmp_Aij=calc_tmpA(tmp_hij);
  
					  tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj));
  
					  search_range=k_search_kappa*fmax(tmp_h,tmp_hj);
  
					  if(tdist<search_range){
						  Real twij,mj,rhoj;
							twij=calc_kernel_wij(tmp_A,tmp_h,tdist);
						  mj=P1[j].m;
						  rhoj=P1[j].rho;
						//   tmp_R+=tmp_hj/P1[j].h0*(mj/rhoj)*twij;
						  tmp_R+=tmp_hj*(mj/rhoj)*twij;
  
			  				tmp_flt+=mj/rhoj*twij;
					  }
				  }
			  }
		  }
	  }
	// P1[i].h=P1[i].h0*tmp_R/tmp_flt;
	P1[i].h=tmp_R/tmp_flt;
	// printf("P1[i].h=%e",P1[i].h);
}
  