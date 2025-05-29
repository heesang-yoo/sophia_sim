////////////////////////////////////////////////////////////////////////
__global__ void kernel_copy_max_velocity(part1*P1,part2*P2,part3*P3,Real*mu)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].p_type==0||P1[i].i_type>=i_type_crt){
		mu[i]=0;
		return;
	}

	mu[i]=sqrt(P1[i].ux*P1[i].ux+P1[i].uy*P1[i].uy+P1[i].uz*P1[i].uz);
}
////////////////////////////////////////////////////////////////////////
__global__ void kernel_copy_max(part1*P1,part2*P2,part3*P3,Real*mrho,Real*mft,Real*mu)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].p_type==0||P1[i].i_type>=i_type_crt){
		mu[i]=0;
		mrho[i]=0;
		mft[i]=0;
		return;
	}

	mu[i]=sqrt(P1[i].ux*P1[i].ux+P1[i].uy*P1[i].uy+P1[i].uz*P1[i].uz);
	// mrho[i]=P1[i].rho;
	mrho[i]=(P1[i].rho-P2[i].rho_ref)/P2[i].rho_ref*100.0;
	mft[i]=P3[i].ftotal;
}
////////////////////////////////////////////////////////////////////////
// __global__ void kernel_copy_max_timestep(part1*P1,part3*P3,Real*mft,Real*mphi)
// {
// 	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
// 	if(i>=k_num_part2) return;
// 	if(P1[i].p_type<=0||P1[i].i_type>=i_type_crt){
// 		mft[i]=0;
// 		mphi[i]=0;
// 		return;
// 	}

// 	mft[i]=P3[i].ftotal;
// 	mphi[i] = P3[i].phi;
// }
__global__ void kernel_copy_max_timestep(part1*P1,part2*P2,part3*P3,Real*dt1,Real*dt2,Real*dt3,Real*dt4,Real*dt5)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].p_type<=0||P1[i].i_type>=i_type_crt){
		dt1[i] = 1.0/0.01;
		dt2[i] = 1.0/0.01;
		dt3[i] = 1.0/0.01;
		dt4[i] = 1.0/0.01;
		dt5[i] = 1.0/0.01;
		return;
	}

	Real hi = P1[i].h;
	Real Di=P1[i].D;
	Real rhoi = P1[i].rho;
	Real tempi=P1[i].temp;
	int_t ptypei=P1[i].p_type;
	Real visi=viscosity(tempi,Di,ptypei);
	Real ci= k_soundspeed;
	if(ptypei==2)	ci=k_soundspeed*sqrt(1000.0/rhoi);
	if(k_flag_timestep_update==1){
		if(k_solver_type==Wcsph){
	dt1[i] = 1.0/(k_CFL*(k_kappa*hi/(ci)));
	// dt2[i] = 1.0/0.01;
	dt2[i] = 1.0/(0.125*rhoi*hi*hi/visi);
	dt3[i] = 1.0/0.01;
	dt4[i] = 1.0/0.01;
	dt5[i] = 1.0/0.01;
		}else if(k_solver_type==Isph){
			Real maxvel = 2.0*sqrt(Gravitational_CONST*0.6);
			Real magvel = sqrt(P1[i].ux*P1[i].ux+P1[i].uy*P1[i].uy);
			dt1[i] = 1.0/(k_CFL_ISPH*(k_kappa*hi/(max(maxvel,magvel))));
			dt2[i] = 1.0/(0.125*rhoi*hi*hi/visi);
		dt3[i] = 1.0/0.01;
		dt4[i] = 1.0/0.01;
		dt5[i] = 1.0/0.01;
	}else if(k_solver_type==Icsph){
		Real maxvel = 2.0*sqrt(Gravitational_CONST*0.6);
		Real magvel = sqrt(P1[i].ux*P1[i].ux+P1[i].uy*P1[i].uy)*(P1[i].p_type==1);
		Real ci= k_soundspeed;
		dt1[i] = 1.0/(k_CFL*(k_kappa*hi/(ci)));
		dt2[i] = 1.0/(0.125*rhoi*hi*hi/visi);
	dt3[i] = 1.0/(k_CFL_ISPH*(k_kappa*hi/(max(maxvel,magvel))));
	// dt3[i] = 1.0/(0.1*(k_kappa*hi/(ci)));
	// dt3[i] = 1.0/(1*(k_kappa*hi/(ci)));
	dt4[i] = 1.0/0.01;
	dt5[i] = 1.0/0.01;
	}
	}else{
		dt1[i] = 1.0/k_dt;
		dt2[i] = 1.0/0.01;
		dt3[i] = 1.0/0.01;
		dt4[i] = 1.0/0.01;
		dt5[i] = 1.0/0.01;
	}

}
float FloatSwap( float f )
{
   union
   {
      float f;
      unsigned char b[4];
      //unsigned char b[8];
   } dat1,dat2;

   dat1.f=f;
   dat2.b[0]=dat1.b[3];
   dat2.b[1]=dat1.b[2];
   dat2.b[2]=dat1.b[1];
   dat2.b[3]=dat1.b[0];
	 /*
   dat2.b[0]=dat1.b[7];
   dat2.b[1]=dat1.b[6];
   dat2.b[2]=dat1.b[5];
   dat2.b[3]=dat1.b[4];
   dat2.b[4]=dat1.b[3];
   dat2.b[5]=dat1.b[2];
   dat2.b[6]=dat1.b[1];
   dat2.b[7]=dat1.b[0];
	 //*/

   return dat2.f;
}
////////////////////////////////////////////////////////////////////////
int IntSwap( int d )
{
   union
   {
      int d;
      unsigned char b[4];
      //unsigned char b[8];
   } dat1,dat2;

   dat1.d=d;
   dat2.b[0]=dat1.b[3];
   dat2.b[1]=dat1.b[2];
   dat2.b[2]=dat1.b[1];
   dat2.b[3]=dat1.b[0];
	 /*
   dat2.b[0]=dat1.b[7];
   dat2.b[1]=dat1.b[6];
   dat2.b[2]=dat1.b[5];
   dat2.b[3]=dat1.b[4];
   dat2.b[4]=dat1.b[3];
   dat2.b[5]=dat1.b[2];
   dat2.b[6]=dat1.b[1];
   dat2.b[7]=dat1.b[0];
	 //*/
   return dat2.d;
}

////////////////////////////////////////////////////////////////////////
void save_plot_fluid_vtk_bin_boundary(part1*P1)
{
	int_t i,nop;//,nob;
	nop=num_part2;
	// nob=number_of_boundaries;
	int_t Nparticle=0;							// number of fluid particles (x>0.00) for 3D PGSFR calculation
	//for(i=0;i<nop;i++) if(P1[i].x>0) Nparticle++;

	for(i=0;i<nop;i++) if(P1[i].p_type==0) Nparticle++;
	printf("test Particle %d\n",Nparticle);

	float val;
		float val1, val2, val3;
	int valt;

	// Filename: It should be series of frame numbers(nameXXX.vtk) for the sake of auto-reading in PARAVIEW.
	char FileName_vtk[256];
	sprintf(FileName_vtk,"./plotdata/boundary_%dstp.vtk",count);
	// If the file already exists,its contents are discarded and create the new one.
	FILE*outFile_vtk;
	outFile_vtk=fopen(FileName_vtk,"w");

	fprintf(outFile_vtk,"# vtk DataFile Version 3.0\n");					// version & identifier: it must be shown.(ver 1.0/2.0/3.0)
	fprintf(outFile_vtk,"Print out results in vtk format\n");			// header: description of file,it never exceeds 256 characters
	fprintf(outFile_vtk,"BINARY\n");														// format of data (ACSII / BINARY)
	fprintf(outFile_vtk,"DATASET POLYDATA\n");										// define DATASET format: 'POLYDATA' is proper to represent SPH particles

	//Define SPH particles---------------------------------------------------------------
	fprintf(outFile_vtk,"POINTS\t%d\tfloat\n",Nparticle);					// define particles position as POINTS
	for(i=0;i<nop;i++){							// print out (x,y,z) coordinates of particles
		//if(P1[i].x>0){
		if(P1[i].p_type==0){
			val=FloatSwap(P1[i].x);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
			val=FloatSwap(P1[i].y);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
			val=FloatSwap(P1[i].z);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"POINT_DATA\t%d\n",Nparticle);

	fprintf(outFile_vtk,"FIELD FieldData\t1\n");


	fprintf(outFile_vtk,"rho\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type==0){
			val=FloatSwap(P1[i].rho);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fclose(outFile_vtk);
}
////////////////////////////////////////////////////////////////////////
void save_plot_moving_vtk_bin(part1*P1)
{
	int_t i,nop;
	nop=num_part2;
	// int_t Nparticle=nop;									// number of fluid particles
	int_t Nparticle=0;									// number of fluid particles
	for(i=0;i<nop;i++) if((P1[i].p_type==9)) Nparticle++;
	// printf("%d test Particle %d\n",tid,Nparticle);

	//*
	float val;
	//int valt;


	// Filename: It should be series of frame numbers(nameXXX.vtk) for the sake of auto-reading in PARAVIEW.
	// If the file already exists,its contents are discarded and create the new one.
	char FileName_vtk[256];
	// sprintf(FileName_vtk,"./plotdata/fluid_%dstp.vtk",count);
	sprintf(FileName_vtk,"./plotdata/moving_%dstp.vtk",count);
	FILE*outFile_vtk;

	outFile_vtk=fopen(FileName_vtk,"w");

	fprintf(outFile_vtk,"# vtk DataFile Version 3.0\n");					// version & identifier: it must be shown.(ver 1.0/2.0/3.0)
	fprintf(outFile_vtk,"Print out results in vtk format\n");			// header: description of file,it never exceeds 256 characters
	fprintf(outFile_vtk,"BINARY\n");														// format of data (ACSII / BINARY)
	fprintf(outFile_vtk,"DATASET POLYDATA\n");										// define DATASET format: 'POLYDATA' is proper to represent SPH particles
	fprintf(outFile_vtk,"POINTS\t%d\tfloat\n",Nparticle);

	// fprintf(outFile_vtk,"%d\n",Nparticle);
	//Define SPH particles---------------------------------------------------------------
	// fprintf(outFile_vtk,"POINTS\t%d\tfloat\n",Nparticle);					// define particles position as POINTS
	for (i=0; i < nop; i++)							// print out (x,y,z) coordinates of particles
	{
		if((P1[i].p_type==9)){
			val=FloatSwap(P1[i].x);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
			val=FloatSwap(P1[i].y);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
			val=FloatSwap(P1[i].z);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"POINT_DATA\t%d\n",Nparticle);
	fprintf(outFile_vtk,"FIELD FieldData\t2\n");

	fprintf(outFile_vtk,"rho\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		// if((P1[i].p_type==2)|(P1[i].p_type==9)){
		if(P1[i].p_type==9){
			val=FloatSwap(P1[i].rho);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}
	fprintf(outFile_vtk,"uy\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		// if((P1[i].p_type==2)|(P1[i].p_type==9)){
		if(P1[i].p_type==9){
			val=FloatSwap(P1[i].uy);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}
	// for (i=0; i < nop; i++)							// print out (x,y,z) coordinates of particles
	// {
	// 	if(P1[i].i_type==1){
	// 		val=FloatSwap(P1[i].pres);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }
	// for (i=0; i < nop; i++)							// print out (x,y,z) coordinates of particles
	// {
	// 	if(P1[i].i_type==1){
	// 		val=FloatSwap(P1[i].rho);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }
	fclose(outFile_vtk);
	//*/
}
////////////////////////////////////////////////////////////////////////
//S.H.Park
void save_restart(part1*P1,part2*P2,part3*P3)
{
	int_t i,nop;//,nob;
	nop=num_part2;
	//int_t Nparticle=nop;									// number of fluid particles

	// Filename: It should be series of frame numbers(nameXXX.vtk) for the sake of auto-reading in PARAVIEW.
	char FileName[256];
	sprintf(FileName,"./plotdata/restart.txt");
	// If the file already exists,its contents are discarded and create the new one.
	FILE*outFile;
	outFile=fopen(FileName,"w");

	fprintf(outFile,"1 2 3 4 5 6 7 8 9 10 11 12 13 26\n");					// version & identifier: it must be shown.(ver 1.0/2.0/3.0)

	//Write data -------------------------------------------------------------------------
	for(i=0;i<nop;i++){
		if(P1[i].i_type<3){
			fprintf(outFile,"%f\t%f\t%f\t",P1[i].x,P1[i].y,P1[i].z);
			fprintf(outFile,"%f\t%f\t%f\t",P1[i].ux,P1[i].uy,P1[i].uz);
			fprintf(outFile,"%e\t%d\t%e\t",P1[i].m,P1[i].p_type,P1[i].h);
			fprintf(outFile,"%f\t%f\t%f\t",P1[i].temp,P1[i].pres,P1[i].rho);
			fprintf(outFile,"%f\t%d\n",P2[i].rho_ref,P1[i].pos);	//check f_total
			// fprintf(outFile,"%f\t%f\t%f\t",P2[i].rho_ref,P3[i].ftotal,P1[i].concn);	//check f_total
			// fprintf(outFile,"%f\t%f\t%d\t",P3[i].cc,P3[i].vis_t,P1[i].ct_boundary);
			// //fprintf(outFile,"%f\t%f\t%d\t%d\t",P3[i].cc,P3[i].vis_t,P1[i].ct_boundary,P3[i].hf_boundary);
			// fprintf(outFile,"%f\t%f\t%f\t%f\t%f\t%f\n",P3[i].lbl_surf,P3[i].drho,P3[i].denthalpy,P3[i].dconcn,P1[i].k_turb,P1[i].e_turb);
	
		}
	}
	fclose(outFile);
}
////////////////////////////////////////////////////////////////////////
void save_vtk_bin_single(part1*P1,part2*P2,part3*P3)
{
	int_t i,nop;//,nob;
	nop=num_part2;
	// nob=number_of_boundaries;
	int_t Nparticle=0;							// number of fluid particles (x>0.00) for 3D PGSFR calculation
	//for(i=0;i<nop;i++) if(P1[i].x>0) Nparticle++;

	for(i=0;i<nop;i++) if(P1[i].p_type>=0) Nparticle++;
	printf("Number of Particles = %d\n\n",Nparticle);

	float val;
	int valt;

	// Filename: It should be series of frame numbers(nameXXX.vtk) for the sake of auto-reading in PARAVIEW.
	char FileName_vtk[256];
	sprintf(FileName_vtk,"./plotdata/fluid_%dstp.vtk",count);
	// If the file already exists,its contents are discarded and create the new one.
	FILE*outFile_vtk;
	outFile_vtk=fopen(FileName_vtk,"w");

	fprintf(outFile_vtk,"# vtk DataFile Version 3.0\n");					// version & identifier: it must be shown.(ver 1.0/2.0/3.0)
	fprintf(outFile_vtk,"Print out results in vtk format\n");			// header: description of file,it never exceeds 256 characters
	fprintf(outFile_vtk,"BINARY\n");														// format of data (ACSII / BINARY)
	fprintf(outFile_vtk,"DATASET POLYDATA\n");										// define DATASET format: 'POLYDATA' is proper to represent SPH particles

	//Define SPH particles---------------------------------------------------------------
	fprintf(outFile_vtk,"POINTS\t%d\tfloat\n",Nparticle);					// define particles position as POINTS
	for(i=0;i<nop;i++){							// print out (x,y,z) coordinates of particles
		//if(P1[i].x>0){
		if(P1[i].p_type>=0){
			val=FloatSwap(P1[i].x);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
			val=FloatSwap(P1[i].y);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
			val=FloatSwap(P1[i].z);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"POINT_DATA\t%d\n",Nparticle);

	fprintf(outFile_vtk,"FIELD FieldData\t%d\n",num_plot_data);

	for (int ccount=0;ccount<num_plot_data;ccount++)
	{
		char data_label[20];
		strcpy(data_label,plot_data[ccount]);

		// buffer_type
		if (!strncmp(data_label,"buffer_type",3)) {
			fprintf(outFile_vtk,"buffer_type\t1\t%d\tint\n",Nparticle);
			for(i=0;i<nop;i++){
				//if(P1[i].x>0){
				if(P1[i].p_type>=0){
					valt=IntSwap(P1[i].buffer_type);
					fwrite((void*)&valt,sizeof(int),1,outFile_vtk);
				}
			}
		}

		if (!strncmp(data_label,"uy",3)) {
			fprintf(outFile_vtk,"D\t1\t%d\tfloat\n",Nparticle);
			for(i=0;i<nop;i++){
					if(P1[i].p_type>=0){
						val=FloatSwap(P1[i].D);
					fwrite((void*)&val,sizeof(float),1,outFile_vtk);
				}
			}
		}

		// concentration
		if (!strncmp(data_label,"concentration",3)) {
			fprintf(outFile_vtk,"concentration\t1\t%d\tfloat\n",Nparticle);
			for(i=0;i<nop;i++){
				//if(P1[i].x>0){
				if(P1[i].p_type>=0){
					val=FloatSwap(P1[i].concentration);
					fwrite((void*)&val,sizeof(float),1,outFile_vtk);
				}
			}
		}

		
		// concentration
		if (!strncmp(data_label,"PPE1",4)) {
			fprintf(outFile_vtk,"PPE1\t1\t%d\tfloat\n",Nparticle);
			for(i=0;i<nop;i++){
				//if(P1[i].x>0){
				if(P1[i].p_type>=0){
					val=FloatSwap(P1[i].PPE1);
					fwrite((void*)&val,sizeof(float),1,outFile_vtk);
				}
			}
		}

		if (!strncmp(data_label,"PPE2",4)) {
			fprintf(outFile_vtk,"PPE2\t1\t%d\tfloat\n",Nparticle);
			for(i=0;i<nop;i++){
				//if(P1[i].x>0){
				if(P1[i].p_type>=0){
					val=FloatSwap(P1[i].PPE2);
					fwrite((void*)&val,sizeof(float),1,outFile_vtk);
				}
			}
		}

				// eli
				if (!strncmp(data_label,"eli",3)) {
					fprintf(outFile_vtk,"eli\t2\t%d\tfloat\n",Nparticle);
					for(i=0;i<nop;i++){
						//if(P1[i].x>0){
						if(P1[i].p_type>=0){
							val=FloatSwap(P1[i].elix);
							fwrite((void*)&val,sizeof(float),1,outFile_vtk);
							val=FloatSwap(P1[i].eliy);
							fwrite((void*)&val,sizeof(float),1,outFile_vtk);
						}
					}
				}
		// i_type
		if (!strncmp(data_label,"i_type",3)) {
			fprintf(outFile_vtk,"i_type\t1\t%d\tint\n",Nparticle);
			for(i=0;i<nop;i++){
				//if(P1[i].x>0){
				if(P1[i].p_type>=0){
					valt=IntSwap(P1[i].i_type);
					fwrite((void*)&valt,sizeof(int),1,outFile_vtk);
				}
			}
		}

		// p_type
		if (!strncmp(data_label,"p_type",3)) {
			fprintf(outFile_vtk,"p_type\t1\t%d\tint\n",Nparticle);
			for(i=0;i<nop;i++){
				//if(P1[i].x>0){
				if(P1[i].p_type>=0){
					valt=IntSwap(P1[i].p_type);
					fwrite((void*)&valt,sizeof(int),1,outFile_vtk);
				}
			}
		}

		
		// detect
		if (!strncmp(data_label,"detect",3)) {
			fprintf(outFile_vtk,"detect\t1\t%d\tint\n",Nparticle);
			for(i=0;i<nop;i++){
				//if(P1[i].x>0){
				if(P1[i].p_type>=0){
					valt=IntSwap(P3[i].lbl_surf);
					fwrite((void*)&valt,sizeof(int),1,outFile_vtk);
				}
			}
		}
// detect
if (!strncmp(data_label,"color",3)) {
	fprintf(outFile_vtk,"color\t1\t%d\tint\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type>=0){
			valt=IntSwap(P3[i].color);
			fwrite((void*)&valt,sizeof(int),1,outFile_vtk);
		}
	}
}
		// p_type
		if (!strncmp(data_label,"ncell",3)) {
			fprintf(outFile_vtk,"ncell\t1\t%d\tint\n",Nparticle);
			for(i=0;i<nop;i++){
				//if(P1[i].x>0){
				if(P1[i].p_type>=0){
					valt=IntSwap(P1[i].ncell);
					fwrite((void*)&valt,sizeof(int),1,outFile_vtk);
				}
			}
		}

		// density
		if (!strncmp(data_label,"rho",3)) {
			fprintf(outFile_vtk,"rho\t1\t%d\tfloat\n",Nparticle);
			for(i=0;i<nop;i++){
				//if(P1[i].x>0){
				// if((P1[i].p_type==2)|(P1[i].p_type==9)){
				if(P1[i].p_type>=0){
					val=FloatSwap(P1[i].rho);
					fwrite((void*)&val,sizeof(float),1,outFile_vtk);
				}
			}
		}

				// density
				if (!strncmp(data_label,"mass",3)) {
					fprintf(outFile_vtk,"m\t1\t%d\tfloat\n",Nparticle);
					for(i=0;i<nop;i++){
						//if(P1[i].x>0){
						// if((P1[i].p_type==2)|(P1[i].p_type==9)){
						if(P1[i].p_type>=0){
							val=FloatSwap(P1[i].m);
							fwrite((void*)&val,sizeof(float),1,outFile_vtk);
						}
					}
				}
		// density
		if (!strncmp(data_label,"vol",3)) {
			fprintf(outFile_vtk,"vol\t1\t%d\tfloat\n",Nparticle);
			for(i=0;i<nop;i++){
				//if(P1[i].x>0){
				// if((P1[i].p_type==2)|(P1[i].p_type==9)){
				if(P1[i].p_type>=0){
					val=FloatSwap(P1[i].vol);
					fwrite((void*)&val,sizeof(float),1,outFile_vtk);
				}
			}
		}
		// density
		if (!strncmp(data_label,"dvol",3)) {
			fprintf(outFile_vtk,"dvol\t1\t%d\tfloat\n",Nparticle);
			for(i=0;i<nop;i++){
				//if(P1[i].x>0){
				// if((P1[i].p_type==2)|(P1[i].p_type==9)){
				if(P1[i].p_type>=0){
					val=FloatSwap(P3[i].dvol);
					fwrite((void*)&val,sizeof(float),1,outFile_vtk);
				}
			}
		}
		if (!strncmp(data_label,"lambda",3)) {
			fprintf(outFile_vtk,"lamb\t1\t%d\tfloat\n",Nparticle);
			for(i=0;i<nop;i++){
					if(P1[i].p_type>=0){
					val=FloatSwap(P3[i].lambda);
					fwrite((void*)&val,sizeof(float),1,outFile_vtk);
				}
			}
		}

		if (!strncmp(data_label,"nvector",3)) {
			fprintf(outFile_vtk,"nv\t3\t%d\tfloat\n",Nparticle);
			for (i=0; i < nop; i++)							// print out (x,y,z) coordinates of particles
			{
				if(P1[i].p_type>=0){
					val=FloatSwap(P3[i].nx);
					fwrite((void*)&val,sizeof(float),1,outFile_vtk);
					val=FloatSwap(P3[i].ny);
					fwrite((void*)&val,sizeof(float),1,outFile_vtk);
					val=FloatSwap(P3[i].nz);
					fwrite((void*)&val,sizeof(float),1,outFile_vtk);
				}
			}
		}

		// ux
		if (!strncmp(data_label,"velocity",2)) {
			fprintf(outFile_vtk,"velocity\t3\t%d\tfloat\n",Nparticle);
			for (i=0; i < nop; i++)							// print out (x,y,z) coordinates of particles
			{
				if(P1[i].p_type>=0){
					val=FloatSwap(P1[i].ux);
					fwrite((void*)&val,sizeof(float),1,outFile_vtk);
					val=FloatSwap(P1[i].uy);
					fwrite((void*)&val,sizeof(float),1,outFile_vtk);
					val=FloatSwap(P1[i].uz);
					fwrite((void*)&val,sizeof(float),1,outFile_vtk);
				}
			}
		}

		if (!strncmp(data_label,"shift",2)) {
			fprintf(outFile_vtk,"shift\t3\t%d\tfloat\n",Nparticle);
			for (i=0; i < nop; i++)							// print out (x,y,z) coordinates of particles
			{
				if(P1[i].p_type>=0){
					val=FloatSwap(P1[i].shiftx);
					fwrite((void*)&val,sizeof(float),1,outFile_vtk);
					val=FloatSwap(P1[i].shifty);
					fwrite((void*)&val,sizeof(float),1,outFile_vtk);
					val=FloatSwap(0.0);
					fwrite((void*)&val,sizeof(float),1,outFile_vtk);
				}
			}
		}

		// uy
		if (!strncmp(data_label,"uanalytic",3)) {
			fprintf(outFile_vtk,"uanalytic\t1\t%d\tfloat\n",Nparticle);
			for(i=0;i<nop;i++){
				//if(P1[i].x>0){
				// if((P1[i].p_type==2)|(P1[i].p_type==9)){
				if(P1[i].p_type>=0){
					val=FloatSwap(P1[i].uanalytic);
					fwrite((void*)&val,sizeof(float),1,outFile_vtk);
				}
			}
		}

		// uz
		if (!strncmp(data_label,"curv",3)) {
			fprintf(outFile_vtk,"curv\t1\t%d\tfloat\n",Nparticle);
			for(i=0;i<nop;i++){
				//if(P1[i].x>0){
				// if((P1[i].p_type==2)|(P1[i].p_type==9)){
				if(P1[i].p_type>=0){
					val=FloatSwap(P3[i].curv);
					fwrite((void*)&val,sizeof(float),1,outFile_vtk);
				}
			}
		}


		// pressure
		if (!strncmp(data_label,"pressure",3)) {
			fprintf(outFile_vtk,"pressure\t1\t%d\tfloat\n",Nparticle);
			for(i=0;i<nop;i++){
				//if(P1[i].x>0){
				// if((P1[i].p_type==2)|(P1[i].p_type==9)){
				if(P1[i].p_type>=0){
					val=FloatSwap(P1[i].pres);
					fwrite((void*)&val,sizeof(float),1,outFile_vtk);
				}
			}
		}

				// pressure
				if (!strncmp(data_label,"dpres",3)) {
					fprintf(outFile_vtk,"dpres\t3\t%d\tfloat\n",Nparticle);
					for(i=0;i<nop;i++){
						//if(P1[i].x>0){
						// if((P1[i].p_type==2)|(P1[i].p_type==9)){
						if(P1[i].p_type>=0){
							val=FloatSwap(P3[i].fpx);
							fwrite((void*)&val,sizeof(float),1,outFile_vtk);
							val=FloatSwap(P3[i].fpy);
							fwrite((void*)&val,sizeof(float),1,outFile_vtk);
							val=FloatSwap(0.0);
							fwrite((void*)&val,sizeof(float),1,outFile_vtk);
						}
					}
				}

		// temp
		if (!strncmp(data_label,"temp",3)) {
			fprintf(outFile_vtk,"temp\t1\t%d\tfloat\n",Nparticle);
			for(i=0;i<nop;i++){
				//if(P1[i].x>0){
				// if((P1[i].p_type==2)|(P1[i].p_type==9)){
				if(P1[i].p_type>=0){
					val=FloatSwap(P1[i].temp);
					fwrite((void*)&val,sizeof(float),1,outFile_vtk);
				}
			}
		}

		// flt_s
		if (!strncmp(data_label,"flt_s",3)) {
			fprintf(outFile_vtk,"flt_s\t1\t%d\tfloat\n",Nparticle);
			for(i=0;i<nop;i++){
				//if(P1[i].x>0){
				// if((P1[i].p_type==2)|(P1[i].p_type==9)){
				if(P1[i].p_type>=0){
					val=FloatSwap(P1[i].flt_s);
					fwrite((void*)&val,sizeof(float),1,outFile_vtk);
				}
			}
		}

		if (!strncmp(data_label,"vorticity",3)) {
			fprintf(outFile_vtk,"vorticity\t1\t%d\tfloat\n",Nparticle);
			for (i=0; i < nop; i++)							// print out (x,y,z) coordinates of particles
			{
				if(P1[i].p_type>=0){
					// val=FloatSwap(P1[i].vortx);
					// fwrite((void*)&val,sizeof(float),1,outFile_vtk);
					// val=FloatSwap(P1[i].vorty);
					// fwrite((void*)&val,sizeof(float),1,outFile_vtk);
					val=FloatSwap(P1[i].vortz);
					fwrite((void*)&val,sizeof(float),1,outFile_vtk);
				}
			}
		}
		if (!strncmp(data_label,"ibm",3)) {
			fprintf(outFile_vtk,"fb\t3\t%d\tfloat\n",Nparticle);
			for(i=0;i<nop;i++){
				//if(P1[i].x>0){
				// if((P1[i].p_type==2)|(P1[i].p_type==9)){
					if(P1[i].p_type>=0){
						val=FloatSwap(P1[i].fbx);
					fwrite((void*)&val,sizeof(float),1,outFile_vtk);
					val=FloatSwap(P1[i].fby);
					fwrite((void*)&val,sizeof(float),1,outFile_vtk);
					val=FloatSwap(P1[i].fbz);
					fwrite((void*)&val,sizeof(float),1,outFile_vtk);
					
				}
			}
		}

		if (!strncmp(data_label,"fbd",3)) {
			fprintf(outFile_vtk,"fbd\t3\t%d\tfloat\n",Nparticle);
			for(i=0;i<nop;i++){
				//if(P1[i].x>0){
				// if((P1[i].p_type==2)|(P1[i].p_type==9)){
					if(P1[i].p_type>=0){
						val=FloatSwap(P3[i].fbdx);
					fwrite((void*)&val,sizeof(float),1,outFile_vtk);
					val=FloatSwap(P3[i].fbdy);
					fwrite((void*)&val,sizeof(float),1,outFile_vtk);
					val=FloatSwap(P3[i].fbdz);
					fwrite((void*)&val,sizeof(float),1,outFile_vtk);
					
				}
			}
		}

		if (!strncmp(data_label,"int",3)) {
			fprintf(outFile_vtk,"u_int\t3\t%d\tfloat\n",Nparticle);
			for(i=0;i<nop;i++){
				//if(P1[i].x>0){
				// if((P1[i].p_type==2)|(P1[i].p_type==9)){
					if(P1[i].p_type>=0){
						val=FloatSwap(P1[i].ux_i);
					fwrite((void*)&val,sizeof(float),1,outFile_vtk);
					val=FloatSwap(P1[i].uy_i);
					fwrite((void*)&val,sizeof(float),1,outFile_vtk);
					val=FloatSwap(P1[i].uz_i);
					fwrite((void*)&val,sizeof(float),1,outFile_vtk);
					
				}
			}
		}


		// if (!strncmp(data_label,"aps",3)) {
		// 	fprintf(outFile_vtk,"aps\t2\t%d\tint\n",Nparticle);
		// 	for(i=0;i<nop;i++){
		// 		//if(P1[i].x>0){
		// 		// if((P1[i].p_type==2)|(P1[i].p_type==9)){
		// 			if(P1[i].p_type>=0){
		// 				valt=IntSwap(P1[i].aps_cond);
		// 				fwrite((void*)&valt,sizeof(int),1,outFile_vtk);
		// 				valt=IntSwap(P1[i].aps_fine);
		// 				fwrite((void*)&valt,sizeof(int),1,outFile_vtk);
		// 		}
		// 	}
		// }

		// // concn
		// if (!strncmp(data_label,"concn",3)) {
		// 	fprintf(outFile_vtk,"concn\t1\t%d\tfloat\n",Nparticle);
		// 	for(i=0;i<nop;i++){
		// 		//if(P1[i].x>0){
		// 		// if((P1[i].p_type==2)|(P1[i].p_type==9)){
		// 		if(P1[i].p_type>=0){
		// 			val=FloatSwap(P1[i].concn);
		// 			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		// 		}
		// 	}
		// }

	}



	fclose(outFile_vtk);
}
