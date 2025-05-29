__device__ void lock(int_t*mutex){
  while(atomicCAS((int_t *)mutex,0,1)!=0);
}

__device__ void unlock(int_t*mutex){
  *mutex=0;
}

// __global__ void KERNEL_set_p_type(part1*P1){
//   int_t i=threadIdx.x+blockIdx.x*blockDim.x;
//   if(i>=k_num_part2) return;
//   Real xi,yi;

//   xi=P1[i].x;
//   yi=P1[i].y;

//   if((yi-0.5)*(yi-0.5)+xi*xi<0.26*0.26){

//     if((yi-0.5)*(yi-0.5)+xi*xi<0.25*0.25) {        // bubble
//       P1[i].p_type=2;
//       P1[i].rho=100.0;
//       P1[i].m=0.01;
//       P1[i].m_ref=0.04;
//     }
//     else if ((P1[i].p_type!=0)&&((yi-0.5)*(yi-0.5)+xi*xi>=0.25*0.25)) {       // water
//       P1[i].p_type=1;
//       P1[i].rho=1000.0;
//       P1[i].m=0.1;
//       P1[i].m_ref=0.4;

//     }
//   }
// }

// __global__ void KERNEL_switch_i_type(part1*P1){
//   int_t i=threadIdx.x+blockIdx.x*blockDim.x;
//   if(i>=k_num_part2) return;
//   if(P1[i].i_type==3) return;

//   Real xi,yi;

//   xi=P1[i].x;
//   yi=P1[i].y;
//   if((xi<-1.0&yi<2.0)||(xi>1.0&yi<2.0)){
//     P1[i].i_type=3;
//   }
//   return;
// }

// __global__ void KERNEL_switch_h(part1*P1){
//   int_t i=threadIdx.x+blockIdx.x*blockDim.x;
//   if(i>=k_num_part2) return;
//   if(P1[i].i_type==3) return;

//   Real href,mref,mi;

//   mi=P1[i].m;
//   mref=P1[i].m_ref;
//   href=P1[i].h_ref;
//   //P1[i].h=href*sqrt(mi/mref);
//   P1[i].h=href;
//   return;
// }

// __global__ void KERNEL_reset_APS_variables(part1*P1){
//   int_t i=threadIdx.x+blockIdx.x*blockDim.x;

//   if(i>=k_num_part2) return;
//   if(P1[i].i_type==3) return;

//   P1[i].aps_cond=0;
//   P1[i].merge_num=0;
//   P1[i].merge_flag=0;
//   P1[i].split_num=0;
//   return;
// }

// __global__ void KERNEL_remove_barrier(part1*P1){
//   int_t i=threadIdx.x+blockIdx.x*blockDim.x;

//   if(i>=k_num_part2) return;
//   if(P1[i].i_type==3) return;

//   if(P1[i].p_type==-1) {
//     P1[i].i_type=3;
//   }
//   return;
// }

void count_APS_buffer(part1*P1,int_t*count_buffer){
  int_t i,nop;
	nop=num_part2;
	int_t Nparticle=0;
	for(i=0;i<nop;i++) if(P1[i].i_type<3) Nparticle++;
  count_buffer[3]=Nparticle-num_part;

  return;
}

void count_buffer_particles(int_t*count_buffer, int_t*dev_count_buffer, int_t*num_buffer_temp, int_t tcount){
  int_t h_num_buffer[1], h_num_buffer_temp[1];

  if(count_buffer[3]!=0) *h_num_buffer=count_buffer[3];
  else{
    cudaMemcpy(count_buffer,dev_count_buffer,4*sizeof(int_t),cudaMemcpyDeviceToHost);
    cudaMemcpy(h_num_buffer_temp,num_buffer_temp,sizeof(int_t),cudaMemcpyDeviceToHost);

    *h_num_buffer=*h_num_buffer_temp;
    *h_num_buffer+=count_buffer[0];
    *h_num_buffer-=count_buffer[1];
    *h_num_buffer-=count_buffer[2];

  }
  cudaMemcpyToSymbol(num_buffer,h_num_buffer,sizeof(int_t),0,cudaMemcpyHostToDevice);
  if((tcount%freq_output)==0) printf("Num_part+buffer : %d\n\n",num_part+*h_num_buffer);
}

// __global__ void KERNEL_smoothing_length(int_t*g_str,int_t*g_end,part1*P1){
//   uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
// 	if(i>=k_num_part2) return;
// 	if(P1[i].i_type>i_type_crt) return;

// 	Real tmp_h=P1[i].h;

// 	int_t icell,jcell;
// 	Real xi,yi;
// 	Real search_range,tmp_A,tmp_R,tmp_flt;

// 	tmp_h=P1[i].h;
// 	tmp_A=calc_tmpA(tmp_h);
// 	search_range=k_search_kappa*tmp_h;	// search range

// 	xi=P1[i].x;
// 	yi=P1[i].y;

// 	// calculate I,J,K in cell
// 	if((k_x_max==k_x_min)){icell=0;}
// 	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
// 	if((k_y_max==k_y_min)){jcell=0;}
// 	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
// 	// out-of-range handling
// 	if(icell<0) icell=0;	if(jcell<0) jcell=0;

//   int_t cell_range=ceil(Cell_division_factor*P1[i].h/P1[i].h_ref);

// 	tmp_R=0.0;
//   tmp_flt=0.0;
// 	for(int_t y=-cell_range;y<=cell_range;y++){
// 		for(int_t x=-cell_range;x<=cell_range;x++){
// 			// int_t k=(icell+x)+k_NI*(jcell+y);
// 			int_t k=idx_cell(icell+x,jcell+y,0);
// 			if (k<0) continue;
// 			if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))) continue;
// 			if(g_str[k]!=cu_memset){
// 				int_t fend=g_end[k];
// 				for(int_t j=g_str[k];j<fend;j++){
// 					Real xj,yj,tdist,tmp_hj,tmp_hij,tmp_Aij;

// 					xj=P1[j].x;
// 					yj=P1[j].y;
// 					tmp_hj=P1[j].h;
// 					tmp_hij=(tmp_h+tmp_hj)/2;
// 					tmp_Aij=calc_tmpA(tmp_hij);

// 					tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj));

// 					if(k_aps_solv) search_range=k_search_kappa*fmax(tmp_h,tmp_hj);

// 					if(tdist<search_range){
// 						Real twij,mj,rhoj;
// 						//if(k_aps_solv) twij=calc_kernel_wij(tmp_Aij,tmp_hij,tdist);
// 					  twij=calc_kernel_wij(tmp_A,tmp_h,tdist);
// 						mj=P1[j].m;
// 						rhoj=P1[j].rho;
//             printf("rhoj:%f\n",rhoj);
// 						tmp_R+=tmp_hj*(mj/rhoj)*twij;

//             tmp_flt+=mj/rhoj*twij;


// 						//printf("rho_ref= %f m= %f wij= %f \n", rho_ref_j, mj, twij);
// 					}
// 				}
// 			}
// 		}
// 	}
// 	// P1[i].rho=rho_ref_i*tmp_R/P1[i].flt_s;
//   P1[i].h=tmp_R/tmp_flt;
//   //printf("filter:%f\n",tmp_flt);
// }


// __global__ void KERNEL_find_miny(part1*P1,Real*min_y){

//   uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
//   if(i>=k_num_part2) return;
//   if((P1[i].p_type==2)||(P1[i].p_type==9)) min_y[i]=P1[i].y;
//   else min_y[i]=10.0;

//   return;
// }

// //-------------------------------------------------------------------------------------------------
// // APR condition
// //-------------------------------------------------------------------------------------------------

// __global__ void KERNEL_APS_condition2D_JI(part1*P1,int_t*aps_num,int_t*g_str,int_t*g_end,Real*aps,int_t tcount){

//   int_t i=threadIdx.x+blockIdx.x*blockDim.x;
//   if(i>=k_num_part2) return;
//   if(P1[i].i_type==3) return;

//   if((tcount==0)&&(P1[i].p_type==0)){       // initial splitting condition
//     P1[i].aps_cond=1;
//     aps_num[0]=1;
//     atomicAdd(&aps_num[1],1);      // count the number of split particles
//     *aps=1.0;
//   }

//   if((tcount==500)&&(P1[i].p_type==0)){       // initial splitting condition
//     P1[i].aps_cond=1;
//     aps_num[0]=1;
//     atomicAdd(&aps_num[1],1);      // count the number of split particles
//     *aps=1.0;
//   }

//   if((P1[i].y>1.0)&&(P1[i].m==P1[i].m_ref)&&(P1[i].x>-1.9)&&(P1[i].x<1.9)){       // Splitting condition 1
//     P1[i].aps_cond=1;
//     aps_num[0]=1;
//     atomicAdd(&aps_num[1],1);      // count the number of split particles
//     *aps=1.0;
//   }

//   if((P1[i].p_type==1)&&(P1[i].y>2.0)&&(P1[i].m>0.2*P1[i].m_ref)&&(P1[i].x>-0.9)&&(P1[i].x<0.9)){     // Splitting condition 2
//     P1[i].aps_cond=1;
//     aps_num[0]=1;
//     atomicAdd(&aps_num[1],1);      // count the number of split particles
//     *aps=1.0;
//   }

//   if((P1[i].p_type==1)&&(P1[i].x>1.0)){       // Cell merging condition 1
//     P1[i].aps_cond=2;
//     aps_num[0]=1;
//     atomicAdd(&aps_num[1],1);      // count the number of split particles
//     *aps=1.0;
//   }

//   if((P1[i].p_type==1)&&(P1[i].x>2.0)){       // Cell merging condtion 2
//     P1[i].aps_cond=3;
//     aps_num[0]=1;
//     atomicAdd(&aps_num[1],1);      // count the number of split particles
//     *aps=1.0;
//   }

//   if((P1[i].p_type==1)&&(P1[i].m<0.25*P1[i].m_ref)&&(P1[i].x<-1.0)){        // particle merging condition 1

//     int_t icell,jcell;
//   	Real xi,yi;
//   	Real mi,rhoi,Vi,merge_range;

//   	xi=P1[i].x;
//   	yi=P1[i].y;
//     mi=P1[i].m;
//     rhoi=P1[i].rho;
//     Vi=mi/rhoi;
//     merge_range=sqrt(3*Vi);

//     if((k_x_max==k_x_min)){icell= 0;}
//     else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
//     if((k_y_max==k_y_min)){jcell=0;}
//     else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}

//     if(icell<0) icell=0;
//     if(jcell<0) jcell=0;

//     int_t cell_range=ceil(Cell_division_factor*P1[i].h/P1[i].h_ref);

//     for(int_t y=-cell_range;y<=cell_range;y++){
//   		for(int_t x=-cell_range;x<=cell_range;x++){
//         int_t kk=idx_cell(icell+x,jcell+y,0);
//         if(kk<0||kk>=k_num_cells-1) continue;
//         if(g_str[kk]!=cu_memset){
//           int_t fend=g_end[kk];

//           for(int_t j=g_str[kk];j<fend;j++){
//             if ((j!=i)&&(P1[j].m==mi)&&(P1[j].p_type==1)){
//               Real xj,yj,zj,jdist;
//               xj=P1[j].x;
//               yj=P1[j].y;
//               jdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj));

//               if(jdist<merge_range){
//                 P1[i].aps_cond=4;
//                 aps_num[0]=1;
//                 *aps=1.0;
//                 atomicAdd(&aps_num[2],1);      // count the number of merging paritcles
//                 return;
//               }
//             }
//           }
//         }
//       }
//     }
//     P1[i].aps_cond=4;
//     P1[i].isolation_count ++;
//   }

//   if((P1[i].p_type==1)&&(P1[i].m<1.0*P1[i].m_ref)&&(P1[i].x<-2.0)){        // particle merging condition 2

//     int_t icell,jcell;
//   	Real xi,yi;
//   	Real mi,rhoi,Vi,merge_range;

//   	xi=P1[i].x;
//   	yi=P1[i].y;
//     mi=P1[i].m;
//     rhoi=P1[i].rho;
//     Vi=mi/rhoi;
//     merge_range=sqrt(3*Vi);

//     if((k_x_max==k_x_min)){icell= 0;}
//     else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
//     if((k_y_max==k_y_min)){jcell=0;}
//     else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}

//     if(icell<0) icell=0;
//     if(jcell<0) jcell=0;

//     int_t cell_range=ceil(Cell_division_factor*P1[i].h/P1[i].h_ref);

//     for(int_t y=-cell_range;y<=cell_range;y++){
//   		for(int_t x=-cell_range;x<=cell_range;x++){
//         int_t kk=idx_cell(icell+x,jcell+y,0);
//         if(kk<0||kk>=k_num_cells-1) continue;
//         if(g_str[kk]!=cu_memset){
//           int_t fend=g_end[kk];

//           for(int_t j=g_str[kk];j<fend;j++){
//             if ((j!=i)&&(P1[j].m==mi)&&(P1[j].p_type==1)){
//               Real xj,yj,zj,jdist;
//               xj=P1[j].x;
//               yj=P1[j].y;
//               jdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj));

//               if(jdist<merge_range){
//                 P1[i].aps_cond=5;
//                 aps_num[0]=1;
//                 *aps=1.0;
//                 atomicAdd(&aps_num[2],1);      // count the number of merging paritcles
//                 return;
//               }
//             }
//           }
//         }
//       }
//     }
//     P1[i].aps_cond=5;
//     P1[i].isolation_count ++;
//   }

// }

//-------------------------------------------------------------------------------------------------
// Splitting & Merging
//-------------------------------------------------------------------------------------------------

__global__ void KERNEL_assign_split_num(part1*P1,int_t*aps_num,int_t*mutex) {
  int_t i=threadIdx.x+blockIdx.x*blockDim.x;

  if(i>k_num_part+*num_buffer) return;
  if(P1[i].i_type==3) return;
  if(P1[i].aps_cond!=1) return;

  if(aps_num[1]>1) {
    lock(mutex);
  }
  if(P1[i].aps_cond==1) {
    atomicAdd(&aps_num[3],1);
    P1[i].split_num=aps_num[3];
  }
  unlock(mutex);
  return;
}

__global__ void KERNEL_particle_splitting2D(part1*P1,int tcount){
  int_t i=threadIdx.x+blockIdx.x*blockDim.x;

  if (i>k_num_part+*num_buffer) return;

  if(P1[i].i_type==3) return;
  if(P1[i].aps_cond!=1) return;
  //if(P1[i].aps_fine!=0) return;

  Real xm,ym,zm;
  Real rhom,mm,Vm;
  int nop;

  xm=P1[i].x;
  ym=P1[i].y;
  zm=P1[i].z;
  rhom=P1[i].rho;
  mm=P1[i].m;
  Vm=P1[i].m/P1[i].rho;
  if((P1[i].aps_cond==1)&&((P1[i].aps_fine==0)||(P1[i].aps_fine==1))){
    // nop=k_num_part2-10*count_buffer[0]-4*(P1[i].split_num);
    nop=k_num_part+*num_buffer+4*(P1[i].split_num);
    int k=0;
    while(k<4){
      P1[nop+k]=P1[i];

      if(k==0){
        P1[nop+k].x=xm-0.25*sqrt(Vm);
        P1[nop+k].y=ym+0.25*sqrt(Vm);
      }
      else if(k==1){
        P1[nop+k].x=xm+0.25*sqrt(Vm);
        P1[nop+k].y=ym+0.25*sqrt(Vm);
      }
      else if(k==2){
        P1[nop+k].x=xm-0.25*sqrt(Vm);
        P1[nop+k].y=ym-0.25*sqrt(Vm);
      }
      else{
        P1[nop+k].x=xm+0.25*sqrt(Vm);
        P1[nop+k].y=ym-0.25*sqrt(Vm);
      }

      P1[nop+k].m=0.25*mm;
      if(k_h_change) P1[nop+k].h=0.5*P1[i].h;
      else P1[nop+k].h=P1[i].h;

      if(P1[i].aps_fine==0) P1[nop+k].aps_fine=1;
      if(P1[i].aps_fine==1) P1[nop+k].aps_fine=2;
      if(P1[i].aps_fine==3) P1[nop+k].aps_fine=1;
      P1[nop+k].aps_cond=0;
      P1[nop+k].split_num=0;

      k++;
    }
    P1[i].i_type=3;
  }
}

// __global__ void KERNEL_assign_merge_num2D(part1* P1,int_t*g_str,int_t*g_end,int_t*aps_num,int_t*mutex){
//   int_t i=threadIdx.x+blockIdx.x*blockDim.x;

//   if (i>k_num_part+*num_buffer) return;
//   if (P1[i].i_type==3) return;
//   if (P1[i].aps_cond!=4&&P1[i].aps_cond!=5) return;

// 	int_t icell,jcell;
//   int_t j;
//   int_t aps_cond;
// 	Real xi,yi;
// 	Real mi,rhoi,Vi,merge_range;

// 	xi=P1[i].x;
// 	yi=P1[i].y;
//   mi=P1[i].m;
//   rhoi=P1[i].rho;
//   aps_cond=P1[i].aps_cond;
//   Vi=mi/rhoi;
//   merge_range=sqrt(3*Vi);

//   if((k_x_max==k_x_min)){icell=0;}
//   else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
//   if((k_y_max==k_y_min)){jcell=0;}
//   else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}

//   if(icell<0) icell=0;
//   if(jcell<0) jcell=0;

//   if(aps_num[2]>1) lock(mutex);

//   if(P1[i].merge_num!=0){
//     unlock(mutex);
//     return;
//   }

//   int_t cell_range=ceil(Cell_division_factor*sqrt(P1[i].h/P1[i].h_ref));

//   for(int_t y=-cell_range;y<=cell_range;y++){
// 		for(int_t x=-cell_range;x<=cell_range;x++){
//       int_t kk=idx_cell(icell+x,jcell+y,0);
//       if(kk<0||kk>=k_num_cells-1) continue;
//       if(g_str[kk]!=cu_memset){
//         int_t fend=g_end[kk];

//         for(j=g_str[kk];j<fend;j++){
//           if ((j!=i)&&(P1[j].aps_cond==aps_cond)&&(P1[j].m==mi)&&(P1[j].merge_num==0)){
//             Real xj,yj,jdist;
//             xj=P1[j].x;
//             yj=P1[j].y;
//             jdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj));
//             if(jdist<merge_range){
//               atomicAdd(&aps_num[4],1);

//               P1[i].merge_num=aps_num[4];
//               P1[j].merge_num=aps_num[4];
//               P1[i].merge_flag=1;

//               unlock(mutex);
//               return;
//             }
//           }
//         }
//       }
//     }
//   }

//   if(P1[i].isolation_count>=5){
//     int_t x=0;
//     int_t y=0;
//     int_t kk=idx_cell(icell+x,jcell+y,0);
//     if(g_str[kk]!=cu_memset){
//       int_t fend=g_end[kk];
//       for(j=g_str[kk];j<fend;j++){
//         if ((j!=i)&&(P1[j].merge_num==0)){
//           Real xj,yj,jdist;
//           if ((P1[j].m<=0.5*P1[j].m_ref)&&(P1[j].p_type==1)){
//             xj=P1[j].x;
//             yj=P1[j].y;
//             jdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj));

//             if(jdist<merge_range){
//               atomicAdd(&aps_num[4],1);
//               P1[i].merge_num=aps_num[4];
//               P1[j].merge_num=aps_num[4];
//               P1[i].merge_flag=1;
//               unlock(mutex);
//               return;
//             }
//           }
//         }
//       }
//     }
//   }
//   unlock(mutex);
//   return;
// }

// __global__ void KERNEL_particle_merging2D(part1* P1,int_t*g_str,int_t*g_end,int_t*aps_num) {
//   int_t i=threadIdx.x+blockIdx.x*blockDim.x;

//   if (i>k_num_part+*num_buffer) return;
//   if (P1[i].i_type==3) return;
//   if (P1[i].merge_flag!=1) return;

// 	int_t icell,jcell,j,k,l;
//   int_t num,num2,nop;
//   Real xi,yi,xj,yj,tmp_h,Vi,merge_range;

//   num=P1[i].merge_num;
//   xi=P1[i].x;
//   yi=P1[i].y;
//   tmp_h=P1[i].h;
//   Vi=P1[i].m/P1[i].rho;
// 	merge_range=sqrt(3*Vi);
//   num2=4*aps_num[3];
//   nop=k_num_part2-2*num;

//   if((k_x_max==k_x_min)){icell= 0;}
//   else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
//   if((k_y_max==k_y_min)){jcell=0;}
//   else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}

//   if(icell<0) icell=0; if(jcell<0) jcell=0;

//   int_t cell_range=ceil(Cell_division_factor*P1[i].h/P1[i].h_ref);

//   for(int_t y=-cell_range;y<=cell_range;y++){
// 		for(int_t x=-cell_range;x<=cell_range;x++){
//       int_t kk=idx_cell(icell+x,jcell+y,0);
//       if(kk<0||kk>=k_num_cells-1) continue;
//       if(g_str[kk]!=cu_memset){
//         int_t fend=g_end[kk];

//         for(j=g_str[kk];j<fend;j++){
//       		if ((j!=i)&&(P1[j].merge_num==num)){
//       			Real jdist;
//       			xj=P1[j].x;
//       			yj=P1[j].y;
//       			jdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj));
//       			if(jdist<merge_range){

//               P1[nop]=P1[i];

//               P1[nop].x=(xi+xj)/2;
//               P1[nop].y=(yi+yj)/2;
//               P1[nop].ux=(P1[i].ux+P1[j].ux)/2;
//               P1[nop].uy=(P1[i].uy+P1[j].uy)/2;
//               P1[nop].m=P1[i].m+P1[j].m;
//               if(k_h_change) P1[nop].h=sqrt(2.0)*tmp_h;
//               else P1[nop].h=tmp_h;
//               P1[nop].temp=(P1[i].temp+P1[j].temp)/2;
//               P1[nop].pres=(P1[i].pres+P1[j].pres)/2;
//               P1[nop].rho=(P1[i].rho+P1[j].rho)/2;
//               P1[nop].enthalpy=(P1[i].enthalpy+P1[j].enthalpy)/2;

//               P1[nop].aps_cond=0;
//               P1[nop].merge_num=0;
//               P1[nop].merge_flag=0;

//               P1[i].i_type=3;
//               P1[j].i_type=3;

//               return;
//             }
//           }
//         }
//       }
//     }
//   }
//   return;
// }

// __global__ void KERNEL_cell_merging2D(part1* P1,int_t*g_str,int_t*g_end,int_t*aps_num) {
//   int_t i=threadIdx.x+blockIdx.x*blockDim.x;    //cell number

//   if (i>k_num_cells-1) return;
//   if (g_str[i]==-1) return;

// 	int_t j,nop,idx_insert,dummy;
//   int_t fend=0;
//   int_t n=0;
//   int_t bflag=0;
//   Real msum=0.0;
//   int_t pnumber[15]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
//   idx_insert=k_num_part2-i;

//   nop=g_end[i]-g_str[i];

//   for(j=g_str[i];j<g_end[i];j++){       // merging condition
//     if(P1[j].aps_cond!=3) return;
//   }

//   for(j=g_str[i];j<g_end[i];j++){
//     P1[j].merge_num=i;
//     if(P1[j].m>=0.8*P1[j].m_ref){
//       pnumber[n]=j;
//       n++;
//       nop--;
//     }
//   }
//   if(nop<=1) return;

//   for(j=g_str[i];j<g_end[i];j++){
//     for(int_t k=0; k<n; k++){
//       if(j==pnumber[k]) {bflag=1; break;}
//       else bflag=0;
//     }
//     if (bflag==1) continue;
//     msum+=P1[j].m;
//     fend=j;
//     if(msum>=1.0*P1[i].m_ref) break;
//   }

//   int_t loop=0;
//   for(j=g_str[i];j<=fend;j++){
//     for(int_t k=0; k<n; k++){
//       if(j==pnumber[k]) {bflag=1; break;}
//       else bflag=0;
//     }
//     if (bflag==1) continue;

//     if(loop==0){          // Initialization
//       P1[idx_insert]=P1[j];
//       P1[idx_insert].x=P1[j].m*P1[j].x/msum;
//       P1[idx_insert].y=P1[j].m*P1[j].y/msum;
//       P1[idx_insert].ux=P1[j].m*P1[j].ux/msum;
//       P1[idx_insert].uy=P1[j].m*P1[j].uy/msum;
//       P1[idx_insert].rho=P1[j].m*P1[j].rho/msum;
//       P1[idx_insert].pres=P1[j].m*P1[j].pres/msum;
//       P1[idx_insert].temp=P1[j].m*P1[j].pres/msum;
//       P1[idx_insert].m=msum;
//     }

//     else{
//       P1[idx_insert].x+=P1[j].m*P1[j].x/msum;
//       P1[idx_insert].y+=P1[j].m*P1[j].y/msum;
//       P1[idx_insert].ux+=P1[j].m*P1[j].ux/msum;
//       P1[idx_insert].uy+=P1[j].m*P1[j].uy/msum;
//       P1[idx_insert].rho+=P1[j].m*P1[j].rho/msum;
//       P1[idx_insert].pres+=P1[j].m*P1[j].pres/msum;
//       P1[idx_insert].temp+=P1[j].m*P1[j].pres/msum;
//     }

//     P1[j].aps_cond=0;
//     P1[j].merge_num=i;
//     P1[j].i_type=3;
//     loop++;
//   }
//   return;
// }

// __global__ void KERNEL_cell_merging2D_2(part1* P1,int_t*g_str,int_t*g_end,int_t*aps_num) {
//   int_t i=threadIdx.x+blockIdx.x*blockDim.x;    //cell number

//   if (i>k_num_cells-1) return;
//   if (g_str[i]==-1) return;

// 	int_t j,nop,idx_insert,dummy;
//   int_t fend=0;
//   int_t n=0;
//   int_t bflag=0;
//   Real msum=0.0;
//   int_t pnumber[15]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
//   idx_insert=k_num_part2-i;

//   nop=g_end[i]-g_str[i];

//   for(j=g_str[i];j<g_end[i];j++){       // merging condition
//     if(P1[j].aps_cond!=2) return;
//   }

//   for(j=g_str[i];j<g_end[i];j++){
//     P1[j].merge_num=i;
//     if(P1[j].m>=0.2*P1[j].m_ref){
//       pnumber[n]=j;
//       n++;
//       nop--;
//     }
//   }
//   if(nop<=1) return;

//   for(j=g_str[i];j<g_end[i];j++){
//     for(int_t k=0; k<n; k++){
//       if(j==pnumber[k]) {bflag=1; break;}
//       else bflag=0;
//     }
//     if (bflag==1) continue;
//     msum+=P1[j].m;
//     fend=j;
//     if(msum>=0.25*P1[i].m_ref) break;
//   }

//   int_t loop=0;
//   for(j=g_str[i];j<=fend;j++){
//     for(int_t k=0; k<n; k++){
//       if(j==pnumber[k]) {bflag=1; break;}
//       else bflag=0;
//     }
//     if (bflag==1) continue;

//     if(loop==0){          // Initialization
//       P1[idx_insert]=P1[j];
//       P1[idx_insert].x=P1[j].m*P1[j].x/msum;
//       P1[idx_insert].y=P1[j].m*P1[j].y/msum;
//       P1[idx_insert].ux=P1[j].m*P1[j].ux/msum;
//       P1[idx_insert].uy=P1[j].m*P1[j].uy/msum;
//       P1[idx_insert].rho=P1[j].m*P1[j].rho/msum;
//       P1[idx_insert].pres=P1[j].m*P1[j].pres/msum;
//       P1[idx_insert].temp=P1[j].m*P1[j].pres/msum;
//       P1[idx_insert].m=msum;
//     }

//     else{
//       P1[idx_insert].x+=P1[j].m*P1[j].x/msum;
//       P1[idx_insert].y+=P1[j].m*P1[j].y/msum;
//       P1[idx_insert].ux+=P1[j].m*P1[j].ux/msum;
//       P1[idx_insert].uy+=P1[j].m*P1[j].uy/msum;
//       P1[idx_insert].rho+=P1[j].m*P1[j].rho/msum;
//       P1[idx_insert].pres+=P1[j].m*P1[j].pres/msum;
//       P1[idx_insert].temp+=P1[j].m*P1[j].pres/msum;
//     }

//     P1[j].aps_cond=0;
//     P1[j].merge_num=i;
//     P1[j].i_type=3;
//     loop++;
//   }
//   return;
// }

// __global__ void KERNEL_APS_condition3D(part1*P1,int_t*aps_num,int_t*g_str,int_t*g_end) {

//   int_t i=threadIdx.x+blockIdx.x*blockDim.x;
//   if(i>=k_num_part2) return;
//   if(P1[i].i_type==3) return;
//   if(P1[i].p_type<1) return;

//   if((P1[i].aps_fine==0)&&(P1[i].x>6.5)){       // splitting condition
//     P1[i].aps_cond=1;
//     aps_num[0]=1;
//     atomicAdd(&aps_num[1],1);      // count the number of split particles
//    }
//   if ((P1[i].aps_fine!=0)&&(P1[i].x<6)){        // merging condition

//     int_t icell,jcell,kcell;
//   	Real xi,yi,zi;
//   	Real mi,rhoi,Vi,merge_range;

//   	xi=P1[i].x;
//   	yi=P1[i].y;
//     zi=P1[i].z;

//     mi=P1[i].m;
//     rhoi=P1[i].rho;
//     Vi=mi/rhoi;
//     merge_range=cbrt(3*Vi);

//     if((k_x_max==k_x_min)){icell=0;}
//     else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
//     if((k_y_max==k_y_min)){jcell=0;}
//     else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
//     if((k_z_max==k_z_min)){kcell=0;}
//     else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}

//     if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

//     for(int_t z=-1;z<=1;z++){
//       for(int_t y=-1;y<=1;y++){
//   		  for(int_t x=-1;x<=1;x++){
//           int_t kk=idx_cell(icell+x,jcell+y,kcell+z);
//           if(kk<0||kk>=k_num_cells-1) continue;
//           if(g_str[kk]!=cu_memset){
//            int_t fend=g_end[kk];

//         	 for(int_t j=g_str[kk];j<fend;j++){
//               if ((j==i)||(P1[j].m!=mi)) continue;
//   					  Real xj,yj,zj,jdist;
//               xj=P1[j].x;
//   					  yj=P1[j].y;
//               zj=P1[j].z;
//               jdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));

//               if(jdist<merge_range){
//                 P1[i].aps_cond=2;
//                 aps_num[0]=1;
//                 atomicAdd(&aps_num[2],1);      // count the number of merging paritcles
//                 return;
//               }
//             }
//           }
//         }
//       }
//     }
//   }
// }

__global__ void KERNEL_particle_splitting3D(part1*P1,int tcount){
    int_t i=threadIdx.x+blockIdx.x*blockDim.x;

    if(i>=k_num_part+*num_buffer) return;
    if(P1[i].i_type==3) return;
    if(P1[i].aps_cond!=1) return;

    Real xm,ym,zm;
    Real rhom,mm,Vm;
    int nop;

    xm=P1[i].x;
    ym=P1[i].y;
    zm=P1[i].z;
    rhom=P1[i].rho;
    mm=P1[i].m;
    Vm=P1[i].m/P1[i].rho;
    nop=k_num_part+*num_buffer+8*(P1[i].split_num);

    int k=0;
    while(k<8){
        P1[nop+k]=P1[i];

        if(k==0){
            P1[nop+k].x=xm-0.25*cbrt(Vm);
            P1[nop+k].y=ym-0.25*cbrt(Vm);
            P1[nop+k].z=zm+0.25*cbrt(Vm);
        }
        else if(k==1){
            P1[nop+k].x=xm-0.25*cbrt(Vm);
            P1[nop+k].y=ym+0.25*cbrt(Vm);
            P1[nop+k].z=zm+0.25*cbrt(Vm);
        }
        else if(k==2){
            P1[nop+k].x=xm+0.25*cbrt(Vm);
            P1[nop+k].y=ym-0.25*cbrt(Vm);
            P1[nop+k].z=zm+0.25*cbrt(Vm);
        }
        else if(k==3){
            P1[nop+k].x=xm+0.25*cbrt(Vm);
            P1[nop+k].y=ym+0.25*cbrt(Vm);
            P1[nop+k].z=zm+0.25*cbrt(Vm);
        }
        else if(k==4){
            P1[nop+k].x=xm-0.25*cbrt(Vm);
            P1[nop+k].y=ym-0.25*cbrt(Vm);
            P1[nop+k].z=zm-0.25*cbrt(Vm);
        }
        else if(k==5){
            P1[nop+k].x=xm-0.25*cbrt(Vm);
            P1[nop+k].y=ym+0.25*cbrt(Vm);
            P1[nop+k].z=zm-0.25*cbrt(Vm);
        }
        else if(k==6){
            P1[nop+k].x=xm+0.25*cbrt(Vm);
            P1[nop+k].y=ym-0.25*cbrt(Vm);
            P1[nop+k].z=zm-0.25*cbrt(Vm);
        }
        else{
            P1[nop+k].x=xm+0.25*cbrt(Vm);
            P1[nop+k].y=ym+0.25*cbrt(Vm);
            P1[nop+k].z=zm-0.25*cbrt(Vm);
        }
        P1[nop+k].m=0.125*mm;
        if(k_h_change) P1[nop+k].h=0.5*P1[i].h;

        P1[nop+k].aps_fine=1;
        P1[nop+k].aps_cond=0;

        k++;
    }
    P1[i].i_type=3;
}

// __global__ void KERNEL_assign_merge_num3D(part1* P1,int_t*g_str,int_t*g_end,int_t*aps_num,int_t*mutex) {
//   int_t i=threadIdx.x+blockIdx.x*blockDim.x;

//   if (i>=k_num_part2) return;
//   if (P1[i].i_type==3) return;
//   if (P1[i].p_type<1) return;
//   if (P1[i].aps_cond!=2) return;

// 	int_t icell,jcell,kcell;
// 	Real xi,yi,zi;
// 	Real mi,rhoi,Vi,merge_range;

// 	xi=P1[i].x;
// 	yi=P1[i].y;
//   zi=P1[i].z;

//   mi=P1[i].m;
//   rhoi=P1[i].rho;
//   Vi=mi/rhoi;
//   merge_range=cbrt(3*Vi);

//   if((k_x_max==k_x_min)){icell=0;}
//   else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
//   if((k_y_max==k_y_min)){jcell=0;}
//   else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
//   if((k_z_max==k_z_min)){kcell=0;}
// 	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}

//   if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

//   if(aps_num[2]>1) {
//     lock(mutex);
//   }

//   for(int_t z=-1;z<=1;z++){
//     for(int_t y=-1;y<=1;y++){
// 		  for(int_t x=-1;x<=1;x++){
//         int_t kk=idx_cell(icell+x,jcell+y,kcell+z);
//         if(kk<0||kk>=k_num_cells-1) continue;
//         if(g_str[kk]!=cu_memset){
//           int_t fend=g_end[kk];

//       	   for(int_t j=g_str[kk];j<fend;j++){
//             if ((j==i)||(P1[j].m!=mi)||(P1[j].merge_num!=0)) continue;
// 				    Real xj,yj,zj,jdist;
//             xj=P1[j].x;
// 				    yj=P1[j].y;
//             zj=P1[j].z;
//             jdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));

//             if(jdist<merge_range){
//               atomicAdd(&aps_num[4],1);
//               P1[i].merge_num=aps_num[4];
//               P1[j].merge_num=aps_num[4];
//               P1[j].merge_flag=1;
//               if(aps_num[2]>1) {
//                 printf("Atomic merging for several merging groups\n\n");
//               }
//               else printf("Particle merging for a single merging group\n\n");
//               unlock(mutex);
//               return;
//             }
//           }
//         }
//       }
//     }
//   }
//   unlock(mutex);
//   return;
// }

// __global__ void KERNEL_particle_merging3D(part1* P1,int_t*g_str,int_t*g_end,int_t*aps_num) {
//   int_t i=threadIdx.x+blockIdx.x*blockDim.x;

//   if (i>=k_num_part2) return;
//   if (P1[i].i_type==3) return;
//   if (P1[i].merge_flag!=1) return;

// 	int_t icell,jcell,kcell;
//   int_t num,num2,nop;
//   Real xi,yi,zi,tmp_h,search_range;

//   num=P1[i].merge_num;
//   xi=P1[i].x;
//   yi=P1[i].y;
//   zi=P1[i].z;
//   tmp_h=P1[i].h;
// 	search_range=k_search_kappa*tmp_h;
//   num2=4*aps_num[3];
//   nop=k_num_part2-num-num2-10;

//   if((k_x_max==k_x_min)){icell=0;}
//   else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
//   if((k_y_max==k_y_min)){jcell=0;}
//   else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
//   if((k_z_max==k_z_min)){kcell=0;}
// 	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}

//   if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

//   for(int_t z=-1;z<=1;z++){
//     for(int_t y=-1;y<=1;y++){
// 		  for(int_t x=-1;x<=1;x++){
//         int_t kk=idx_cell(icell+x,jcell+y,kcell+z);
//         if(kk<0||kk>=k_num_cells-1) continue;
//         if(g_str[kk]!=cu_memset){
//           int_t fend=g_end[kk];

//       	   for(int_t j=g_str[kk];j<fend;j++){
//              if (j==i) continue;
// 					   Real xj,yj,zj,jdist;
//              xj=P1[j].x;
// 			       yj=P1[j].y;
//              zj=P1[j].z;
//             jdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));

//             if((jdist<search_range)&&(P1[j].merge_num==num)){

//               P1[nop]=P1[i];

//               P1[nop].x=(xi+xj)/2;
//               P1[nop].y=(yi+yj)/2;
//               P1[nop].z=(zi+zj)/2;
//               P1[nop].ux=(P1[i].ux+P1[j].ux)/2;
//               P1[nop].uy=(P1[i].uy+P1[j].uy)/2;
//               P1[nop].uz=(P1[i].uz+P1[j].uz)/2;
//               P1[nop].m=P1[i].m+P1[j].m;
//               P1[nop].h=cbrt(2.0)*tmp_h;
//               P1[nop].temp=(P1[i].temp+P1[j].temp)/2;
//               P1[nop].pres=(P1[i].pres+P1[j].pres)/2;

//               P1[nop].rho=(P1[i].rho+P1[j].rho)/2;
//               P1[nop].flt_s=(P1[i].flt_s+P1[j].flt_s)/2;
//               P1[nop].w_dx=(P1[i].w_dx+P1[j].w_dx)/2;
//               P1[nop].enthalpy=(P1[i].enthalpy+P1[j].enthalpy);
//               P1[nop].concn=(P1[i].concn+P1[j].concn)/2;

//               P1[nop].grad_rhox=(P1[i].grad_rhox+P1[j].grad_rhox)/2;
//               P1[nop].grad_rhoy=(P1[i].grad_rhoy+P1[j].grad_rhoy)/2;
//               P1[nop].grad_rhoz=(P1[i].grad_rhoz+P1[j].grad_rhoz)/2;
//               P1[nop].k_turb=(P1[i].k_turb+P1[j].k_turb)/2;
//               P1[nop].e_turb=(P1[i].e_turb+P1[j].e_turb)/2;

//               if (P1[i].aps_fine==1) P1[nop].aps_fine=2;
//               else if (P1[i].aps_fine==2) P1[nop].aps_fine=3;
//               else if (P1[i].aps_fine==3) P1[nop].aps_fine=0;
//               P1[nop].aps_cond=0;
//               P1[nop].merge_num=0;
//               P1[nop].merge_flag=0;

//               P1[j].i_type=3;
//               P1[i].i_type=3;

//               atomicSub(&k_num_part2,1);

//               return;
//             }
//           }
//         }
//       }
//     }
//   }
//   return;
// }

// __global__ void KERNEL_store_min(part1*P1,Real*x_min_storage,Real*y_min_storage){

//   uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
//   if(i>k_num_part2) return;

//   if((P1[i].p_type==2)&&(P1[i].i_type==1)){
//     x_min_storage[i]=P1[i].x;
//     y_min_storage[i]=P1[i].y;
//   }
//   else {
//     x_min_storage[i]=100.0;
//     y_min_storage[i]=100.0;
//   }
//   return;
// }

// __global__ void KERNEL_store_max(part1*P1,Real*x_max_storage,Real*y_max_storage){

//   uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
//   if(i>k_num_part2) return;
//   if(P1[i].i_type==3) return;

//   if(P1[i].p_type==2){
//     x_max_storage[i]=P1[i].x;
//     y_max_storage[i]=P1[i].y;
//   }
//   else {
//     x_max_storage[i]=-100.0;
//     y_max_storage[i]=-100.0;
//   }
//   return;
// }


// // ====================================================================================

// // __global__ void KERNEL_assign_merge_num2D(part1* P1,int_t*g_str,int_t*g_end,int_t*aps_num,int_t*mutex){
// //   int_t i=threadIdx.x+blockIdx.x*blockDim.x;
// //
// //   if (i>k_num_part+*num_buffer) return;
// //   if (P1[i].i_type==3) return;
// //   if (P1[i].p_type<1) return;
// //   if (P1[i].aps_cond!=2) return;
// //
// // 	int_t icell,jcell;
// //   int_t j,k,l;
// // 	Real xi,yi;
// // 	Real mi,rhoi,Vi,merge_range;
// //
// // 	xi=P1[i].x;
// // 	yi=P1[i].y;
// //
// //   mi=P1[i].m;
// //   rhoi=P1[i].rho;
// //   Vi=mi/rhoi;
// //   merge_range=sqrt(3*Vi);
// //
// //   if((k_x_max==k_x_min)){icell=0;}
// //   else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
// //   if((k_y_max==k_y_min)){jcell=0;}
// //   else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
// //
// //   if(icell<0) icell=0;
// //   if(jcell<0) jcell=0;
// //
// //   if(aps_num[2]>1) lock(mutex);
// //   __syncthreads();
// //
// //   if(P1[i].merge_num!=0){
// //     unlock(mutex);
// //     return;
// //   }
// //
// //   for(int_t y=-1;y<=1;y++){
// // 		for(int_t x=-1;x<=1;x++){
// //       int_t kk=idx_cell(icell+x,jcell+y,0);
// //       if(kk<0||kk>=k_num_cells-1) continue;
// //       if(g_str[kk]!=cu_memset){
// //         int_t fend=g_end[kk];
// //
// //         for(j=g_str[kk];j<fend;j++){
// //           if ((j!=i)&&(P1[j].aps_cond==2)&&(P1[j].merge_num==0)){
// //             Real xj,yj,jdist;
// //             xj=P1[j].x;
// //             yj=P1[j].y;
// //             jdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj));
// //             if(jdist<merge_range) break;
// //           }
// //         }
// //
// //         for(k=j+1;k<fend;k++){
// //           if ((k!=i)&&(P1[k].aps_cond==2)&&(P1[k].merge_num==0)){
// //             Real xk,yk,kdist;
// //             xk=P1[k].x;
// //             yk=P1[k].y;
// //             kdist=sqrt((xi-xk)*(xi-xk)+(yi-yk)*(yi-yk));
// //             if(kdist<merge_range) break;
// //           }
// //         }
// //
// //         for(l=k+1;l<fend;l++){
// //           if ((l!=i)&&(P1[l].aps_cond==2)&&(P1[l].merge_num==0)){
// //             Real xl,yl,ldist;
// //             xl=P1[l].x;
// //             yl=P1[l].y;
// //             ldist=sqrt((xi-xl)*(xi-xl)+(yi-yl)*(yi-yl));
// //
// //             if(ldist<merge_range){
// //               atomicAdd(&aps_num[4],1);
// //
// //               P1[i].merge_num=aps_num[4];
// //               P1[j].merge_num=aps_num[4];
// //               P1[k].merge_num=aps_num[4];
// //               P1[l].merge_num=aps_num[4];
// //               P1[i].merge_flag=1;
// //
// //               unlock(mutex);
// //               return;
// //             }
// //           }
// //         }
// //       }
// //     }
// //   }
// //   unlock(mutex);
// //   return;
// // }
// //
// // __global__ void KERNEL_particle_merging2D(part1* P1,int_t*g_str,int_t*g_end,int_t*aps_num) {
// //   int_t i=threadIdx.x+blockIdx.x*blockDim.x;
// //
// //   if (i>k_num_part+*num_buffer) return;
// //   if (P1[i].i_type==3) return;
// //   if (P1[i].merge_flag!=1) return;
// //
// // 	int_t icell,jcell,j,k,l;
// //   int_t num,num2,nop;
// //   Real xi,yi,xj,yj,xk,yk,xl,yl,tmp_h,Vi,merge_range;
// //
// //   num=P1[i].merge_num;
// //   xi=P1[i].x;
// //   yi=P1[i].y;
// //   tmp_h=P1[i].h;
// //   Vi=P1[i].m/P1[i].rho;
// // 	merge_range=sqrt(3*Vi);
// //   num2=4*aps_num[3];
// //   nop=k_num_part2-2*num;
// //
// //   if((k_x_max==k_x_min)){icell= 0;}
// //   else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
// //   if((k_y_max==k_y_min)){jcell=0;}
// //   else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
// //
// //   if(icell<0) icell=0; if(jcell<0) jcell=0;
// //
// //   for(int_t y=-1;y<=1;y++){
// // 		for(int_t x=-1;x<=1;x++){
// //       int_t kk=idx_cell(icell+x,jcell+y,0);
// //       if(kk<0||kk>=k_num_cells-1) continue;
// //       if(g_str[kk]!=cu_memset){
// //         int_t fend=g_end[kk];
// //
// //         for(j=g_str[kk];j<fend;j++){
// //       		if ((j!=i)&&(P1[j].merge_num==num)){
// //       			Real jdist;
// //       			xj=P1[j].x;
// //       			yj=P1[j].y;
// //       			jdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj));
// //       			if(jdist<merge_range) break;
// //       		}
// //       	}
// //
// //       	for(k=j+1;k<fend;k++){
// //       		if ((k!=i)&&(P1[k].merge_num==num)){
// //       			Real kdist;
// //       			xk=P1[k].x;
// //       			yk=P1[k].y;
// //       			kdist=sqrt((xi-xk)*(xi-xk)+(yi-yk)*(yi-yk));
// //       			if(kdist<merge_range) break;
// //       		}
// //       	}
// //
// //       	for(l=k+1;l<fend;l++){
// //       		if ((l!=i)&&(P1[l].merge_num==num)){
// //       			Real ldist;
// //       			xl=P1[l].x;
// //       			yl=P1[l].y;
// //       			ldist=sqrt((xi-xl)*(xi-xl)+(yi-yl)*(yi-yl));
// //             if(ldist<merge_range){
// //               P1[nop]=P1[i];
// //
// //               P1[nop].x=(xi+xj+xk+xl)/4;
// //               P1[nop].y=(yi+yj+yk+yl)/4;
// //               P1[nop].ux=(P1[i].ux+P1[j].ux+P1[k].ux+P1[l].ux)/4;
// //               P1[nop].uy=(P1[i].uy+P1[j].uy+P1[k].uy+P1[l].uy)/4;
// //               P1[nop].m=P1[i].m+P1[j].m+P1[k].m+P1[l].m;
// //               P1[nop].h=2.0*tmp_h;
// //               P1[nop].temp=(P1[i].temp+P1[j].temp+P1[k].temp+P1[l].temp)/4;
// //               P1[nop].pres=(P1[i].pres+P1[j].pres+P1[k].pres+P1[l].pres)/4;
// //               P1[nop].rho=(P1[i].rho+P1[j].rho+P1[k].rho+P1[l].rho)/4;
// //               P1[nop].enthalpy=(P1[i].enthalpy+P1[j].enthalpy+P1[k].enthalpy+P1[l].enthalpy)/4;
// //
// //               P1[nop].aps_fine=0;
// //               P1[nop].aps_cond=0;
// //               P1[nop].merge_num=0;
// //               P1[nop].merge_flag=0;
// //
// //               P1[i].i_type=3;
// //               P1[j].i_type=3;
// //               P1[k].i_type=3;
// //               P1[l].i_type=3;
// //
// //               return;
// //             }
// //           }
// //         }
// //       }
// //     }
// //   }
// //   return;
// // }

// // __global__ void KERNEL_APS_condition2D_EC(part1*P1,int_t*aps_num,int_t*g_str,int_t*g_end){
// //
// //   int_t i=threadIdx.x+blockIdx.x*blockDim.x;
// //   if(i>=k_num_part2) return;
// //   if(P1[i].i_type==3) return;
// //   if(P1[i].p_type!=1) return;
// //
// //   if((P1[i].aps_fine==0)&&(P1[i].x>8.5)){       // splitting condition
// //     P1[i].aps_cond=1;
// //     aps_num[0]=1;
// //     atomicAdd(&aps_num[1],1);      // count the number of split particles
// //    }
// //   if((P1[i].aps_fine!=0)&&(P1[i].x<8)){        // merging condition
// //
// //     int_t icell,jcell;
// //   	Real xi,yi;
// //   	Real mi,rhoi,Vi,merge_range;
// //
// //   	xi=P1[i].x;
// //   	yi=P1[i].y;
// //
// //     mi=P1[i].m;
// //     rhoi=P1[i].rho;
// //     Vi=mi/rhoi;
// //     merge_range=sqrt(2*Vi);
// //
// //     if((k_x_max==k_x_min)){icell= 0;}
// //     else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
// //     if((k_y_max==k_y_min)){jcell=0;}
// //     else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
// //
// //     if(icell<0) icell=0;
// //     if(jcell<0) jcell=0;
// //
// //     for(int_t y=-1;y<=1;y++){
// //   		for(int_t x=-1;x<=1;x++){
// //         int_t kk=idx_cell(icell+x,jcell+y,0);
// //         if(kk<0||kk>=k_num_cells-1) continue;
// //         if(g_str[kk]!=cu_memset){
// //          int_t fend=g_end[kk];
// //
// //          for(int_t j=g_str[kk];j<fend;j++){
// //             if ((j==i)||(P1[j].m!=mi)) continue;
// //             Real xj,yj,zj,jdist;
// //             xj=P1[j].x;
// //             yj=P1[j].y;
// //             jdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj));
// //
// //             if(jdist<merge_range){
// //               for(int_t k=g_str[kk];k<fend;k++){
// //                 if ((k==i)||(k==j)||(P1[k].m!=mi)) continue;
// //                 Real xk,yk,zk,kdist;
// //                 xk=P1[k].x;
// //                 yk=P1[k].y;
// //                 kdist=sqrt((xi-xk)*(xi-xk)+(yi-yk)*(yi-yk));
// //
// //                 if(kdist<merge_range){
// //                   for(int_t l=g_str[kk];l<fend;l++){
// //                     if ((l==i)||(l==j)||(l==k)||(P1[l].m!=mi)) continue;
// //                     Real xl,yl,zl,ldist;
// //                     xl=P1[l].x;
// //                     yl=P1[l].y;
// //                     ldist=sqrt((xi-xl)*(xi-xl)+(yi-yl)*(yi-yl));
// //
// //                     if(ldist<merge_range){
// //                       P1[i].aps_cond=2;
// //                       aps_num[0]=1;
// //                       atomicAdd(&aps_num[2],1);      // count the number of merging paritcles
// //                       return;
// //                     }
// //                   }
// //                 }
// //               }
// //             }
// //           }
// //         }
// //       }
// //     }
// //   }
// //   return;
// // }

// // __global__ void KERNEL_assign_merge_num2D_EC(part1* P1,int_t*g_str,int_t*g_end,int_t*aps_num,int_t*mutex){
// //   int_t i=threadIdx.x+blockIdx.x*blockDim.x;
// //
// //   if(i>=k_num_part2) return;
// //   if(P1[i].i_type==3) return;
// //   if(P1[i].p_type!=1) return;
// //   if(P1[i].aps_cond!=2) return;
// //
// //   int_t icell,jcell;
// // 	Real xi,yi;
// // 	Real mi,rhoi,Vi,merge_range;
// //
// // 	xi=P1[i].x;
// // 	yi=P1[i].y;
// //
// //   mi=P1[i].m;
// //   rhoi=P1[i].rho;
// //   Vi=mi/rhoi;
// //   merge_range=sqrt(2*Vi);
// //
// //   if((k_x_max==k_x_min)){icell= 0;}
// //   else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
// //   if((k_y_max==k_y_min)){jcell=0;}
// //   else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
// //
// //   if(icell<0) icell=0;
// //   if(jcell<0) jcell=0;
// //
// //   if(aps_num[2]>1) {
// //     lock(mutex);
// //   }
// //
// //   for(int_t y=-1;y<=1;y++){
// // 		for(int_t x=-1;x<=1;x++){
// //       int_t kk=idx_cell(icell+x,jcell+y,0);
// //       if(kk<0||kk>=k_num_cells-1) continue;
// //       if(g_str[kk]!=cu_memset){
// //        int_t fend=g_end[kk];
// //
// //        for(int_t j=g_str[kk];j<fend;j++){
// //           if ((j==i)||(P1[j].m!=mi)||(P1[j].merge_num!=0)) continue;
// //           Real xj,yj,zj,jdist;
// //           xj=P1[j].x;
// //           yj=P1[j].y;
// //           jdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj));
// //
// //           if(jdist<merge_range){
// //             for(int_t k=g_str[kk];k<fend;k++){
// //               if ((k==i)||(k==j)||(P1[k].m!=mi)||(P1[k].merge_num!=0)) continue;
// //               Real xk,yk,zk,kdist;
// //               xk=P1[k].x;
// //               yk=P1[k].y;
// //               kdist=sqrt((xi-xk)*(xi-xk)+(yi-yk)*(yi-yk));
// //
// //               if(kdist<merge_range){
// //                 for(int_t l=g_str[kk];l<fend;l++){
// //                   if ((l==i)||(l==j)||(l==k)||(P1[l].m!=mi)||(P1[l].merge_num!=0)) continue;
// //                   Real xl,yl,zl,ldist;
// //                   xl=P1[l].x;
// //                   yl=P1[l].y;
// //                   ldist=sqrt((xi-xl)*(xi-xl)+(yi-yl)*(yi-yl));
// //
// //                   if(ldist<merge_range){
// //                     atomicAdd(&aps_num[4],1);
// //                     P1[i].merge_num=aps_num[4];
// //                     P1[j].merge_num=aps_num[4];
// //                     P1[k].merge_num=aps_num[4];
// //                     P1[l].merge_num=aps_num[4];
// //                     P1[i].merge_flag=1;
// //                     unlock(mutex);
// //                     return;
// //                   }
// //                 }
// //               }
// //             }
// //           }
// //         }
// //       }
// //     }
// //   }
// //   unlock(mutex);
// //   return;
// // }
// //
// // __global__ void KERNEL_particle_merging2D_EC(part1* P1,int_t*g_str,int_t*g_end,int_t*aps_num_part,int_t*aps_num) {
// //   int_t i=threadIdx.x+blockIdx.x*blockDim.x;
// //
// //   if (i>=k_num_part2) return;
// //   if (P1[i].i_type==3) return;
// //   if (P1[i].merge_flag!=1) return;
// //
// // 	int_t icell,jcell;
// //   int_t num,num2,nop;
// //   Real xi,yi,tmp_h,search_range;
// //
// //   num=P1[i].merge_num;
// //   xi=P1[i].x;
// //   yi=P1[i].y;
// //   tmp_h=P1[i].h;
// // 	search_range=k_search_kappa*tmp_h;
// //   num2=4*aps_num[3];
// //   nop=k_num_part2-2*num-num2-10;
// //
// //   if((k_x_max==k_x_min)){icell= 0;}
// //   else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
// //   if((k_y_max==k_y_min)){jcell=0;}
// //   else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
// //
// //   if(icell<0) icell=0;
// //   if(jcell<0) jcell=0;
// //
// //   for(int_t y=-1;y<=1;y++){
// // 		for(int_t x=-1;x<=1;x++){
// //       int_t kk=idx_cell(icell+x,jcell+y,0);
// //       if(kk<0||kk>=k_num_cells-1) continue;
// //       if(g_str[kk]!=cu_memset){
// //        int_t fend=g_end[kk];
// //
// //        for(int_t j=g_str[kk];j<fend;j++){
// //           if (j==i) continue;
// //           Real xj,yj,zj,jdist;
// //           xj=P1[j].x;
// //           yj=P1[j].y;
// //           jdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj));
// //
// //           if((jdist<search_range)&&(P1[j].merge_num==num)){
// //             for(int_t k=g_str[kk];k<fend;k++){
// //               if ((k==i)||(k==j)) continue;
// //               Real xk,yk,zk,kdist;
// //               xk=P1[k].x;
// //               yk=P1[k].y;
// //               kdist=sqrt((xi-xk)*(xi-xk)+(yi-yk)*(yi-yk));
// //
// //               if((kdist<search_range)&&(P1[k].merge_num==num)){
// //                 for(int_t l=g_str[kk];l<fend;l++){
// //                   if ((l==i)||(l==j)||(l==k)) continue;
// //                   Real xl,yl,zl,ldist;
// //                   xl=P1[l].x;
// //                   yl=P1[l].y;
// //                   ldist=sqrt((xi-xl)*(xi-xl)+(yi-yl)*(yi-yl));
// //
// //                   if((ldist<search_range)&&(P1[l].merge_num==num)){
// //                     double uix,uiy,ujx,ujy,ukx,uky,ulx,uly;
// //                     uix=P1[i].ux; uiy=P1[i].uy;
// //                     ujx=P1[j].ux; ujy=P1[j].uy;
// //                     ukx=P1[k].ux; uky=P1[k].uy;
// //                     ulx=P1[l].ux; uly=P1[l].uy;
// //
// //                     double a=0.5*(uix+ujx+ukx+ulx);
// //                     double b=0.5*(uiy+ujy+uky+uly);
// //                     double c=0.5*((uix*uix)+(uiy*uiy)+(ujx*ujx)+(ujy*ujy)+(ukx*ukx)+(uky*uky)+(ulx*ulx)+(uly*uly));
// //
// //                     for(int_t m=0;m<=1;m++){
// //                       P1[nop+m]=P1[i];
// //
// //                       if(m==0){
// //                         P1[nop+m].x=(xi+xj)/2;
// //                         P1[nop+m].y=(yi+yj)/2;
// //                         P1[nop+m].ux=a/2-(b/2)*sqrt((2*c/(a*a+b*b))-1);
// //                         if (b==0) P1[nop+m].uy=sqrt(c/2-a*a/4);
// //                         else P1[nop+m].uy=((a*a+b*b)/(2*b))-(a/b)*P1[nop+m].ux;
// //                       }
// //                       else if(m==1){
// //                         P1[nop+m].x=(xk+xl)/2;
// //                         P1[nop+m].y=(yk+yl)/2;
// //                         P1[nop+m].ux=a-P1[nop].ux;
// //                         P1[nop+m].uy=b-P1[nop].uy;
// //                       }
// //
// //                       P1[nop+m].m=2*P1[i].m;
// //                       P1[nop+m].h=sqrt(2.0)*tmp_h;
// //                       P1[nop+m].temp=(P1[i].temp+P1[j].temp+P1[k].temp+P1[l].temp)/4;
// //                       P1[nop+m].pres=(P1[i].pres+P1[j].pres+P1[k].pres+P1[l].pres)/4;
// //
// //                       P1[nop+m].rho=(P1[i].rho+P1[j].rho+P1[k].rho+P1[l].rho)/4;
// //                       P1[nop+m].flt_s=(P1[i].flt_s+P1[j].flt_s+P1[k].flt_s+P1[l].flt_s)/4;
// //                       P1[nop+m].w_dx=(P1[i].w_dx+P1[j].w_dx+P1[k].w_dx+P1[l].w_dx)/4;
// //                       P1[nop+m].enthalpy=(P1[i].enthalpy+P1[j].enthalpy+P1[k].enthalpy+P1[l].enthalpy)/4;
// //                       P1[nop+m].concn=(P1[i].concn+P1[j].concn+P1[k].concn+P1[l].concn)/4;
// //
// //                       P1[nop+m].grad_rhox=(P1[i].grad_rhox+P1[j].grad_rhox+P1[k].grad_rhox+P1[l].grad_rhox)/4;
// //                       P1[nop+m].grad_rhoy=(P1[i].grad_rhoy+P1[j].grad_rhoy+P1[k].grad_rhoy+P1[l].grad_rhoy)/4;
// //                       P1[nop+m].grad_rhoz=(P1[i].grad_rhoz+P1[j].grad_rhoz+P1[k].grad_rhoz+P1[l].grad_rhoz)/4;
// //                       P1[nop+m].k_turb=(P1[i].k_turb+P1[j].k_turb+P1[k].k_turb+P1[l].k_turb)/4;
// //                       P1[nop+m].e_turb=(P1[i].e_turb+P1[j].e_turb+P1[k].e_turb+P1[l].e_turb)/4;
// //
// //                       if (P1[i].aps_fine==1) P1[nop+m].aps_fine=2;
// //                       else if (P1[i].aps_fine==2) P1[nop+m].aps_fine=0;
// //                       P1[nop+m].aps_cond=0;
// //                       P1[nop+m].merge_num=0;
// //                       P1[nop+m].merge_flag=0;
// //                     }
// //                     //if((abs(P1[nop].x-P1[nop+1].x)<0.01)&&(abs(P1[nop].y-P1[nop+1].y)<0.01)) P1[nop+1].x+=0.03;
// //
// //                     P1[i].i_type=3;
// //                     P1[j].i_type=3;
// //                     P1[k].i_type=3;
// //                     P1[l].i_type=3;
// //
// //                     atomicSub(aps_num_part,2);
// //                     return;
// //                   }
// //                 }
// //               }
// //             }
// //           }
// //         }
// //       }
// //     }
// //   }
// //   return;
// // }
