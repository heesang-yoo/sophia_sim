// ====================================================================================
// 2D APR
// ====================================================================================

__global__ void KERNEL_APS_condition2D_init(part1*P1,int_t*aps_num,Real*aps){
  int_t i=threadIdx.x+blockIdx.x*blockDim.x;    //cell number

  if (i>k_num_part2-1) return;

  if(P1[i].p_type<=0) {
    P1[i].aps_cond=1;
    atomicAdd(&aps_num[1],1);      // initial setup
    *aps=1.0;
  }
  else if(P1[i].x<0.015 || P1[i].x>0.085 || P1[i].y<0.015 || P1[i].y>0.085){
    P1[i].aps_cond=1;
    atomicAdd(&aps_num[1],1);      // initial setup
    *aps=1.0;
  }
}

__global__ void KERNEL_APS_condition3D_init(part1*P1,int_t*aps_num,Real*aps){
  int_t i=threadIdx.x+blockIdx.x*blockDim.x;    //cell number

  if (i>k_num_part2-1) return;

  if(P1[i].p_type<=0|P1[i].x<0) {
    P1[i].aps_cond=1;
    atomicAdd(&aps_num[1],1);      // initial setup
    *aps=1.0;
  }
}

__global__ void KERNEL_APS_condition2D_cell(part1*P1,int_t*aps_cell,int_t*g_str,int_t*g_end,Real*aps,int_t tcount){
  int_t i=threadIdx.x+blockIdx.x*blockDim.x;    //cell number

  if (i>k_num_cells-1) return;
  if (g_str[i]==-1) return;

  Real x_cell=0;
  Real y_cell=0;
  int_t nop=g_end[i]-g_str[i];

  for(int_t j=g_str[i];j<g_end[i];j++){
    if(P1[j].p_type==1) break;
    if(j==g_end[i]-1) return;
  }

  for(int_t j=g_str[i];j<g_end[i];j++){
    x_cell+=P1[j].x/nop;
    y_cell+=P1[j].y/nop;
  }

  if(x_cell<-5 || x_cell>5 || y_cell<0 || y_cell>5) aps_cell[i]=1;       // Splitting condition
  if(x_cell<4.99 && x_cell>-4.99 && y_cell>0.01 && y_cell<4.99) aps_cell[i]=-1;      // Merging condition
}

__global__ void KERNEL_cell_APR_2D(part1*P1,part2*P2,int_t*g_str,int_t*g_end,int_t*aps_cell,int_t tcount){
  int_t i=threadIdx.x+blockIdx.x*blockDim.x;    //cell number

  if (i>k_num_cells-1) return;
  if (g_str[i]==-1) return;

  int_t j,k,nop,idx_insert;
  int_t fend=0;
  int_t n=0;
  int_t bflag=0;
  Real msum=0.0;
  int_t pnumber[2]={0,0};


  if(aps_cell[i]<0){          // Cell merging
    for(j=g_str[i];j<g_end[i];j++){
      if(P1[j].m<=0.8*P1[j].m0 && P1[j].p_type==1){
        pnumber[n]=j;
        n++;
        if(n==2) break;
      }
    }
    if (n!=2) return;

    idx_insert=k_num_part2+aps_cell[i];
    j=pnumber[0];
    k=pnumber[1];
    Real msum, mj, mk;
    mj=P1[j].m;
    mk=P1[k].m;
    msum=mj+mk;

    if (msum>P1[j].m0*1.2) return;

    P1[idx_insert]=P1[j];

    P1[idx_insert].m=msum;
    P1[idx_insert].x=(mj*P1[j].x+mk*P1[k].x)/msum;
    P1[idx_insert].y=(mj*P1[j].y+mk*P1[k].y)/msum;
    P1[idx_insert].ux=(mj*P1[j].ux+mk*P1[k].ux)/msum;
    P1[idx_insert].uy=(mj*P1[j].uy+mk*P1[k].uy)/msum;
    P1[idx_insert].pres=(mj*P1[j].pres+mk*P1[k].pres)/msum;
    P1[idx_insert].rho=(mj*P1[j].rho+mk*P1[k].rho)/msum;
    if(k_h_change) P1[idx_insert].h=P1[idx_insert].h0*sqrt(msum/P1[idx_insert].m0);

    P1[idx_insert].aps_cond=0;

    P1[j].i_type=3;
    P1[k].i_type=3;

    return;
  }

  if(aps_cell[i]>0){          // Cell splitting

    for(j=g_str[i];j<g_end[i];j++){
      if (P1[j].p_type!=1 || P1[j].m<0.4*P1[j].m0) continue;
      P1[j].split_num=aps_cell[i];

      if(P1[j].p_type==1 && P1[j].m>0.75*P1[j].m0){
        // printf("mean vel:%f count:%d \n",P1[j].cell_prop,tcount);
        idx_insert=k_num_part+*num_buffer+4*aps_cell[i];

        Real xm=P1[j].x;
        Real ym=P1[j].y;
        Real Vm=P1[j].m/P1[j].rho;

        k=0;
        while(k<4){
          P1[idx_insert+k]=P1[j];

          if(k==0){
            P1[idx_insert+k].x=xm-0.25*sqrt(Vm);
            P1[idx_insert+k].y=ym+0.25*sqrt(Vm);
          }
          else if(k==1){
            P1[idx_insert+k].x=xm+0.25*sqrt(Vm);
            P1[idx_insert+k].y=ym+0.25*sqrt(Vm);
          }
          else if(k==2){
            P1[idx_insert+k].x=xm-0.25*sqrt(Vm);
            P1[idx_insert+k].y=ym-0.25*sqrt(Vm);
          }
          else{
            P1[idx_insert+k].x=xm+0.25*sqrt(Vm);
            P1[idx_insert+k].y=ym-0.25*sqrt(Vm);
          }

          P1[idx_insert+k].m=0.25*P1[j].m;
          if(k_h_change) P1[idx_insert+k].h=0.5*P1[j].h;

          P1[idx_insert+k].aps_cond=0;

          k++;
        }
        P1[j].i_type=3;
        return;
      }

      else if(P1[j].p_type==1 && P1[j].m>0.4*P1[j].m0){
        // printf("mean vel:%f count:%d \n",P1[j].cell_prop,tcount);
        idx_insert=k_num_part+*num_buffer+4*aps_cell[i];

        Real xm=P1[j].x;
        Real ym=P1[j].y;
        Real Vm=P1[j].m/P1[j].rho;

        k=0;
        while(k<2){
          P1[idx_insert+k]=P1[j];

          if(k==0){
            P1[idx_insert+k].x=xm-0.5*sqrt(Vm);
            P1[idx_insert+k].y=ym+0.5*sqrt(Vm);
          }
          else{
            P1[idx_insert+k].x=xm+0.5*sqrt(Vm);
            P1[idx_insert+k].y=ym-0.5*sqrt(Vm);
          }

          P1[idx_insert+k].m=0.5*P1[j].m;
          if(k_h_change) P1[idx_insert+k].h=sqrt(0.5)*P1[j].h;

          P1[idx_insert+k].aps_cond=0;

          k++;
        }
        P1[j].i_type=3;
        return;
      }
    }
  }
}

// ====================================================================================
// 3D APR
// ====================================================================================

__global__ void KERNEL_APS_3D_init(part1*P1,int_t*aps_num,Real*aps){
  int_t i=threadIdx.x+blockIdx.x*blockDim.x;    //cell number

  if (i>k_num_part2-1) return;

  if(P1[i].p_type<=0) {
    P1[i].aps_cond=1;
    atomicAdd(&aps_num[1],1);      // initial setup
    *aps=1.0;
  }
}

__global__ void KERNEL_APS_condition3D_cell(part1*P1,int_t*aps_cell,int_t*g_str,int_t*g_end,Real*aps,int_t tcount){
  int_t i=threadIdx.x+blockIdx.x*blockDim.x;    //cell number

  if (i>k_num_cells-1) return;
  if (g_str[i]==-1) return;

  Real vor,x,y,z;
  vor=x=y=z=0.0;

  int_t nop=g_end[i]-g_str[i];

  for(int_t j=g_str[i];j<g_end[i];j++){
    if(P1[j].p_type==1) break;
    if(j==g_end[i]-1) return;
  }

  // for(int_t j=g_str[i];j<g_end[i];j++){
  //   x+=P1[j].x/nop;
  //   y+=P1[j].y/nop;
  //   z+=P1[j].z/nop;
  //   vor+=abs(P1[j].vortz)/nop;
  // }

  //if(x<0 && y<0 && z<0) aps_cell[i]=1;       // Splitting condition
  //if(x>0 && y>0 && z>0) aps_cell[i]=1;      // Merging condition

  for(int_t j=g_str[i];j<g_end[i];j++){
    x+=P1[j].x/nop;
  }

  for(int_t j=g_str[i];j<g_end[i];j++){
    if(x<0) aps_cell[i]=1;
  }
}

__global__ void KERNEL_cell_APR_3D(part1*P1,part2*P2,int_t*g_str,int_t*g_end,int_t*aps_cell,int_t tcount){
  int_t i=threadIdx.x+blockIdx.x*blockDim.x;    //cell number

  if (i>k_num_cells-1) return;
  if (g_str[i]==-1) return;

  int_t j,k,nop,idx_insert;
  int_t fend=0;
  int_t n=0;
  int_t bflag=0;
  Real msum=0.0;
  int_t pnumber[2]={0,0};

  if(aps_cell[i]<0){          // Cell merging
    for(j=g_str[i];j<g_end[i];j++){
      if(P1[j].m<2.0*P1[j].m0 && P1[j].p_type==1){
        pnumber[n]=j;
        n++;
        if(n==2) break;
      }
    }
    if (n!=2) return;

    idx_insert=k_num_part2+aps_cell[i];
    j=pnumber[0];
    k=pnumber[1];
    Real msum, mj, mk;
    mj=P1[j].m;
    mk=P1[k].m;
    msum=mj+mk;

    if (msum>P1[j].m0*2.0) return;

    P1[idx_insert]=P1[j];

    P1[idx_insert].m=msum;
    P1[idx_insert].x=(mj*P1[j].x+mk*P1[k].x)/msum;
    P1[idx_insert].y=(mj*P1[j].y+mk*P1[k].y)/msum;
    P1[idx_insert].ux=(mj*P1[j].ux+mk*P1[k].ux)/msum;
    P1[idx_insert].uy=(mj*P1[j].uy+mk*P1[k].uy)/msum;
    P1[idx_insert].pres=(mj*P1[j].pres+mk*P1[k].pres)/msum;
    P1[idx_insert].rho=(mj*P1[j].rho+mk*P1[k].rho)/msum;
    if(k_h_change) P1[idx_insert].h=P1[idx_insert].h0*cbrt(msum/P1[idx_insert].m0);

    P1[idx_insert].aps_cond=0;

    P1[j].i_type=3;
    P1[k].i_type=3;

    return;
  }

  if(aps_cell[i]>0){          // Cell splitting

    for(j=g_str[i];j<g_end[i];j++){
      if (P1[j].p_type!=1 || P1[j].m<0.4*P1[j].m0) continue;
      P1[j].split_num=aps_cell[i];

      if(P1[j].p_type==1 && P1[j].m>0.8*P1[j].m0-1e-6){

        idx_insert=k_num_part+*num_buffer+8*aps_cell[i];

        Real xm=P1[j].x;
        Real ym=P1[j].y;
        Real zm=P1[j].z;
        Real Vm=P1[j].m/P1[j].rho;

        k=0;
        while(k<8){
          P1[idx_insert+k]=P1[j];
          P2[idx_insert+k]=P2[j];

          if(k==0){
            P1[idx_insert+k].x=xm+0.25*cbrt(Vm);
            P1[idx_insert+k].y=ym+0.25*cbrt(Vm);
            P1[idx_insert+k].z=zm+0.25*cbrt(Vm);
          }
          else if(k==1){
            P1[idx_insert+k].x=xm+0.25*cbrt(Vm);
            P1[idx_insert+k].y=ym+0.25*cbrt(Vm);
            P1[idx_insert+k].z=zm-0.25*cbrt(Vm);
          }
          else if(k==2){
            P1[idx_insert+k].x=xm+0.25*cbrt(Vm);
            P1[idx_insert+k].y=ym-0.25*cbrt(Vm);
            P1[idx_insert+k].z=zm+0.25*cbrt(Vm);
          }
          else if(k==3){
            P1[idx_insert+k].x=xm+0.25*cbrt(Vm);
            P1[idx_insert+k].y=ym-0.25*cbrt(Vm);
            P1[idx_insert+k].z=zm-0.25*cbrt(Vm);
          }
          else if(k==4){
            P1[idx_insert+k].x=xm-0.25*cbrt(Vm);
            P1[idx_insert+k].y=ym+0.25*cbrt(Vm);
            P1[idx_insert+k].z=zm+0.25*cbrt(Vm);
          }
          else if(k==5){
            P1[idx_insert+k].x=xm-0.25*cbrt(Vm);
            P1[idx_insert+k].y=ym+0.25*cbrt(Vm);
            P1[idx_insert+k].z=zm-0.25*cbrt(Vm);
          }
          else if(k==6){
            P1[idx_insert+k].x=xm-0.25*cbrt(Vm);
            P1[idx_insert+k].y=ym-0.25*cbrt(Vm);
            P1[idx_insert+k].z=zm+0.25*cbrt(Vm);
          }
          else if(k==7){
            P1[idx_insert+k].x=xm-0.25*cbrt(Vm);
            P1[idx_insert+k].y=ym-0.25*cbrt(Vm);
            P1[idx_insert+k].z=zm-0.25*cbrt(Vm);
          }

          P1[idx_insert+k].m=0.125*P1[j].m;
          if(k_h_change) P1[idx_insert+k].h=0.5*P1[j].h;
          else P1[idx_insert+k].h=P1[j].h;

          if(P1[j].aps_fine==0) P1[idx_insert+k].aps_fine=1;
          if(P1[j].aps_fine==1) P1[idx_insert+k].aps_fine=2;
          if(P1[j].aps_fine==3) P1[idx_insert+k].aps_fine=1;
          P1[idx_insert+k].aps_cond=0;

          k++;
        }
        P1[j].i_type=3;
        return;
      }

      else if(P1[j].p_type==1 && P1[j].m>0.4*P1[j].m0){
        // printf("mean vel:%f count:%d \n",P1[j].cell_prop,tcount);
        idx_insert=k_num_part+*num_buffer+8*aps_cell[i];

        Real xm=P1[j].x;
        Real ym=P1[j].y;
        Real zm=P1[j].z;
        Real Vm=P1[j].m/P1[j].rho;

        k=0;
        while(k<4){
          P1[idx_insert+k]=P1[j];

          if(k==0){
            P1[idx_insert+k].x=xm+0.25*cbrt(Vm);
            P1[idx_insert+k].z=zm+0.25*cbrt(Vm);
          }
          if(k==1){
            P1[idx_insert+k].x=xm+0.25*cbrt(Vm);
            P1[idx_insert+k].z=zm-0.25*cbrt(Vm);
          }
          if(k==1){
            P1[idx_insert+k].x=xm-0.25*cbrt(Vm);
            P1[idx_insert+k].z=zm+0.25*cbrt(Vm);
          }
          else{
            P1[idx_insert+k].x=xm-0.25*cbrt(Vm);
            P1[idx_insert+k].z=ym-0.25*cbrt(Vm);
          }

          P1[idx_insert+k].m=0.25*P1[j].m;
          if(k_h_change) P1[idx_insert+k].h=sqrt(0.5)*P1[j].h;
          else P1[idx_insert+k].h=P1[j].h;

          P1[idx_insert+k].aps_cond=0;

          k++;
        }
        P1[j].i_type=3;
        return;
      }
    }
  }
}

// ====================================================================================
// etc
// ====================================================================================

__global__ void Kernel_change_ptype(part1*P1){
  int_t i=threadIdx.x+blockIdx.x*blockDim.x;

  if (i>k_num_part2-1) return;

  Real xi,yi,xcom,ycom,r;

  xi=P1[i].x;
  yi=P1[i].y;

  xcom=0.05;
  ycom=0.05;
  r=0.0049;

  if (((ycom-yi)*(ycom-yi)+(xcom-xi)*(xcom-xi))<r*r) P1[i].p_type=-2;
  else if (((ycom-yi)*(ycom-yi)+(xcom-xi)*(xcom-xi))<=1.1*r*r) P1[i].p_type=1;

  return;
}

__global__ void Kernel_variable_smoothing_length2D(int_t*g_str,int_t*g_end,part1*TP1,part1*P1,int_t tcount){
  int_t i=threadIdx.x+blockIdx.x*blockDim.x;

  if(i>k_num_part2-1) return;
  if(P1[i].i_type>i_type_crt) return;
  if(P1[i].p_type<1) return;

  int_t icell,jcell;
  Real xi,yi,hi,h_ref_i;
  Real search_range,tmp_A,tmp_R,tmp_flt,tmp_RR;

  tmp_R=tmp_flt=0.0;

  hi=P1[i].h;
  h_ref_i=P1[i].h0;
  tmp_A=calc_tmpA(hi);
  search_range=k_search_kappa*hi;	// search range

  xi=P1[i].x;
  yi=P1[i].y;

  // calculate I,J,K in cell
  if((k_x_max==k_x_min)){icell=0;}
  else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
  if((k_y_max==k_y_min)){jcell=0;}
  else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
  // out-of-range handling
  if(icell<0) icell=0;	if(jcell<0) jcell=0;

  int_t cell_range=ceil(Cell_division_factor*P1[i].h/P1[i].h0);

  tmp_RR=0.0;
  for(int_t y=-cell_range;y<=cell_range;y++){
    for(int_t x=-cell_range;x<=cell_range;x++){
      // int_t k=(icell+x)+k_NI*(jcell+y);
      int_t k=idx_cell(icell+x,jcell+y,0);
      if (k<0) continue;
      if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))) continue;
      if(g_str[k]!=cu_memset){
        int_t fend=g_end[k];
        for(int_t j=g_str[k];j<fend;j++){
          Real xj,yj,mj,hj,rhoj,tdist,ptypej;

          xj=P1[j].x;
          yj=P1[j].y;
          mj=P1[j].m;
          hj=P1[j].h;
          rhoj=P1[j].rho;
          ptypej=P1[j].p_type;
          tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj));

          if(k_aps_solv) search_range=k_search_kappa*fmax(hi,hj);

          if(tdist<search_range){
            Real twij;

            twij=calc_kernel_wij(tmp_A,P1[j].h0,tdist);
            tmp_R+=twij;
            tmp_RR+=(ptypej==1)*hj*mj/rhoj*twij;
            tmp_flt+=(ptypej==1)*mj/rhoj*twij;
          }
        }
      }
    }
  }
  if (tcount%5000==0)TP1[i].h=1.5*(sqrt(1/tmp_R));
  else TP1[i].h=tmp_RR/tmp_flt;
}

__global__ void Kernel_variable_smoothing_length3D(int_t*g_str,int_t*g_end,part1*P1,int_t tcount){
  int_t i=threadIdx.x+blockIdx.x*blockDim.x;

  if (i>k_num_part2-1) return;
  if(P1[i].i_type>i_type_crt) return;
  if(P1[i].p_type<1) return;

  int_t icell,jcell,kcell;
  Real xi,yi,zi,hi;
  Real search_range,tmp_A,tmp_R,tmp_RR,tmp_flt;

  tmp_R=tmp_RR=tmp_flt=0.0;

  hi=P1[i].h;
  tmp_A=calc_tmpA(hi);
  search_range=k_search_kappa*hi;	// search range

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
  if(icell<0) icell=0;	if(jcell<0) jcell=0;  if(kcell<0) kcell=0;

  int_t cell_range=ceil(Cell_division_factor*P1[i].h/P1[i].h0);

  for(int_t z=-cell_range;z<=cell_range;z++){
    for(int_t y=-cell_range;y<=cell_range;y++){
      for(int_t x=-cell_range;x<=cell_range;x++){
        // int_t k=(icell+x)+k_NI*(jcell+y);
        int_t k=idx_cell(icell+x,jcell+y,kcell+z);
        if (k<0) continue;
        if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;
        if(g_str[k]!=cu_memset){
          int_t fend=g_end[k];
          for(int_t j=g_str[k];j<fend;j++){
            Real xj,yj,zj,hj,mj,rhoj,tdist;

            xj=P1[j].x;
            yj=P1[j].y;
            zj=P1[j].z;
            mj=P1[j].m;
            hj=P1[j].h;
            rhoj=P1[j].rho;

            tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj))+1e-20;

            if(k_aps_solv) search_range=k_search_kappa*fmax(hi,hj);

            if(tdist<search_range){
              Real twij;
              int_t ptypej=P1[j].p_type;

              twij=calc_kernel_wij(tmp_A,hi,tdist);
              tmp_R+=twij;
              tmp_RR+=hj*mj/rhoj*twij;
              tmp_flt+=mj/rhoj*twij;
            }
          }
        }
      }
    }
  }
  if(tcount%5000==0)P1[i].h=1.5*(cbrt(1/tmp_R));
  else P1[i].h=tmp_RR/tmp_flt;
}

void Cell_index(int_t*APS_cell){
  int_t j=1;
  int_t k=-1;

  for (int_t i=0; i<=num_cells; i++){
    if(APS_cell[i]>0){
      APS_cell[i]=j;
      j++;
    }
    if(APS_cell[i]<0){
      APS_cell[i]=k;
      k--;
    }
  }
  return;
}

void APR_init(part1*dev_P1,Real*aps,int_t tcount){
  int_t*mutex, *dev_APS_num;																		// mutex, aps num 등 APS 관련 변수 선언 및 메모리 할당

  cudaMalloc((void**)&mutex,sizeof(int_t));
  cudaMemset(mutex,0,sizeof(int_t));
  cudaMalloc((void**)&dev_APS_num,5*sizeof(int_t));
  cudaMemset(dev_APS_num,0,5*sizeof(int_t));

  dim3 b,t;

  b.x=(num_part2-1)/t.x+1;
  if(dim==2) KERNEL_APS_condition2D_init<<<b,t>>>(dev_P1,dev_APS_num,aps);
  if(dim==3) KERNEL_APS_condition3D_init<<<b,t>>>(dev_P1,dev_APS_num,aps);
  cudaDeviceSynchronize();

  t.x=1;   // 원자연산이 필요한 경우 t.x 값을 1로 설정
  b.x=(num_part2-1)/t.x+1;
  KERNEL_assign_split_num<<<b,t>>>(dev_P1,dev_APS_num,mutex);   // APS split 입자에 split_num 값 부여
  cudaDeviceSynchronize();

  t.x=128;
  b.x=(num_part2-1)/t.x+1;
  if(dim==2) KERNEL_particle_splitting2D<<<b,t>>>(dev_P1,tcount);   // Particle splitting
  if(dim==3) KERNEL_particle_splitting3D<<<b,t>>>(dev_P1,tcount);   // Particle splitting
  cudaDeviceSynchronize();

  cudaFree(mutex);
  cudaFree(dev_APS_num);

}

__global__ void KERNEL_TEST(part1*P1,part3*P3){
  int_t i=threadIdx.x+blockIdx.x*blockDim.x;    //cell number

  if (i>k_num_part2-1) return;
  if (P1[i].p_type!=1) return;

  Real x,y,z;
  x=P1[i].x;
  y=P1[i].y;

  if(P3[i].lbl_surf==1) P1[i].cell_prop=1.0;

}
