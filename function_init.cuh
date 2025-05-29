// int_t type variable define
#define		solver_type							vii[0]		// solver type: WCSPH/PCISPH/DFSPH
#define		dim											vii[1]		// dimension
#define		prop_table							vii[2]		// property_table(Yes/NO)
#define		kernel_type							vii[3]		// kernel type
#define		flt_type								vii[4]		// filter type
#define		rho_type								vii[5]		// density calcuation type
#define		time_type								vii[6]		// time stepping type
#define		scheme									vii[7]		// fluid type
#define		open_boundary						vii[8]		// open boundary (YES / NO)
#define		simulation_type					vii[9]		// simulation type(single_phase / two_phase)
#define		freq_cell								vii[10]		// cell initialization frequency
#define		flag_z_index						vii[11]		// flag for z-indexing
#define		flag_timestep_update		vii[12]		// flag for varying timestep update
#define		nb_cell_type						vii[13]		// neighbor cell type
#define		freq_filt								vii[14]		// filtering frequency
#define		freq_mass_sum						vii[15]		// mass summation frequency
#define		freq_temp								vii[16]		// temperature filtering frequency
#define		freq_output							vii[17]		// output frequency
#define		fp_solve								vii[18]		// solve pressure force ?
#define		fv_solve								vii[19]		// sovle viscous force ?
#define		fva_solve								vii[20]		// solve artificial viscous force ?
#define		fg_solve								vii[21]		// solve gravity force ?
#define		fs_solve								vii[22]		// solve surface tension force ?
#define		fb_solve								vii[23]		// solve boundary force ?
#define		con_solve								vii[24]		// solve conduction?
#define		boussinesq_solve				vii[25]		// solve boussinesq approximation based natural convection?
#define		interface_solve					vii[26]		// solve interface sharpness force?
#define		surf_model							vii[27]		// surface tension model
#define		xsph_solve							vii[28]		// solve xsph ?
#define		kgc_solve								vii[29]		// solve kernel gradient correction ?
#define		delSPH_solve						vii[30]		// solve delta-SPH ?
#define		delSPH_model						vii[31]		// delta SPH model
#define		pst_solve								vii[32]		// solve particle shifting ?
#define		turbulence_model				vii[33]		// turbulence model	(by esk)
#define		concn_solve							vii[34]		// concentration diffusion model(PSH)
//
//psh:: ISPH input
#define		minIteration						vii[35]		// minimum number of PCISPH iteration
#define		maxIteration						vii[36]		// maximum number of PCISPH iteration
//
//solution variables
#define		nb_cell_number					vii[37]
#define		num_part								vii[38]
#define		number_of_boundaries		vii[39]
#define		NI											vii[40]
#define		NJ											vii[41]
#define		NK											vii[42]
#define		count										vii[43]
#define		num_cells								vii[44]
#define		ngpu										vii[45]
#define		calc_area								vii[46]
#define		num_p2p									vii[47]
#define		num_part2								vii[48]
#define		penetration_solve				vii[49]
#define		multi_type						vii[50]
#define   noslip_bc								vii[51]
#define		IPF_kernel_type							vii[52]		// kernel type
#define		num_part3								vii[53]
#define		num_marker								vii[54]
#define   aps_solv                vii[55]
#define   structure_solv                vii[56]
#define 	h_change								vii[57]
#define 	viscosity_type							vii[58]
#define		timematchingstep										vii[59]		// time-step(s)

//
// Real type variable define
#define		kappa										vif[0]		// k in k*h
#define		p_ref										vif[1]		// reference pressure(for EOS)
#define		gamma										vif[2]		// gamma
#define		dt											vif[3]		// time-step(s)
#define		time										vif[4]		// time(s)
#define		time_end								vif[5]
// margin for simulation range
#define		Xmargin_m								vif[6]
#define		Xmargin_p								vif[7]
#define		Ymargin_m								vif[8]
#define		Ymargin_p								vif[9]
#define		Zmargin_m								vif[10]
#define		Zmargin_p								vif[11]
#define		u_limit									vif[12]
#define		c_xsph									vif[13]		// coefficient for XSPH
#define		c_repulsive							vif[14]		// coefficient for repulsive boundary force
//
//psh:: ISPH input
#define		drho_th									vif[15]		// density convergence criterion
#define		dp_th										vif[16]		// pressure convergence criterion
#define		p_relaxation						vif[17]		// relaxation factor for PCISPH pressure
//
//solution variables
#define		x_min										vif[18]
#define		x_max										vif[19]
#define		y_min										vif[20]
#define		y_max										vif[21]
#define		z_min										vif[22]
#define		z_max										vif[23]
#define		h_max									vif[24]
#define		soundspeed							vif[25]
#define		rho0_eos								vif[26]
//
#define		search_incr_factor			vif[27]
#define		search_kappa						vif[28]
#define		dcell										vif[29]
#define		ball_vel								vif[30]
#define		dt_structure											vif[31]		// time-step(s)
#define		time_output											vif[32]		// time-step(s)
#define		time2											vif[33]		// time-step(s)
#define		dt_2											vif[34]		// time-step(s)
#define		CFL											vif[35]		// time-step(s)
#define		CFL_ISPH											vif[36]		// time-step(s)

////////////////////////////////////////////////////////////////////////
// int_t type variable define
#define		k_solver_type							k_vii[0]		// solver type: WCSPH/PCISPH/DFSPH
#define		k_dim											k_vii[1]		// dimension
#define		k_prop_table							k_vii[2]		// property table (YES/NO)
#define		k_kernel_type							k_vii[3]		// kernel type
#define		k_flt_type								k_vii[4]		// filter type
#define		k_rho_type								k_vii[5]		// density calcuation type
#define		k_time_type								k_vii[6]		// time stepping type
#define		k_scheme									k_vii[7]		// fluid type
#define		k_open_boundary						k_vii[8]		// open boundary (YES / NO)
#define		k_simulation_type					k_vii[9]		// simulation type(single_phase / two_phase)
#define		k_freq_cell								k_vii[10]		// cell initialization frequency
#define		k_flag_z_index						k_vii[11]		// flag for z-indexing
#define		k_flag_timestep_update		k_vii[12]		// flag for varying timestep update
#define		k_nb_cell_type						k_vii[13]		// neighbor cell type
#define		k_freq_filt								k_vii[14]		// filtering frequency
#define		k_freq_mass_sum						k_vii[15]		// mass summation frequency
#define		k_freq_temp								k_vii[16]		// temperature filtering frequency
#define		k_freq_output							k_vii[17]		// output frequency
#define		k_fp_solve								k_vii[18]		// solve pressure force ?
#define		k_fv_solve								k_vii[19]		// sovle viscous force ?
#define		k_fva_solve								k_vii[20]		// solve artificial viscous force ?
#define		k_fg_solve								k_vii[21]		// solve gravity force ?
#define		k_fs_solve								k_vii[22]		// solve surface tension force ?
#define		k_fb_solve								k_vii[23]		// solve boundary force ?
#define		k_con_solve								k_vii[24]		// solve conduction?
#define		k_boussinesq_solve				k_vii[25]		// solve boussinesq approximation based natural convection?
#define		k_interface_solve					k_vii[26]		// solve interface sharpness force?
#define		k_surf_model							k_vii[27]		// surface tension model
#define		k_xsph_solve							k_vii[28]		// solve xsph ?
#define		k_kgc_solve								k_vii[29]		// solve kernel gradient correction ?
#define		k_delSPH_solve						k_vii[30]		// solve delta-SPH ?
#define		k_delSPH_model						k_vii[31]		// delta SPH model
#define		k_pst_solve								k_vii[32]		// solve particle shifting ?
#define		k_turbulence_model				k_vii[33]		// turbulence model	(by esk)
#define		k_concn_solve							k_vii[34]		// concentration diffusion model(PSH)
//
//psh:: ISPH input
#define		k_minIteration						k_vii[35]		// minimum number of PCISPH iteration
#define		k_maxIteration						k_vii[36]		// maximum number of PCISPH iteration
//
//solution variables
#define		k_nb_cell_number					k_vii[37]
#define		k_num_part			k_vii[38]
#define		k_number_of_boundaries		k_vii[39]
#define		k_NI											k_vii[40]
#define		k_NJ											k_vii[41]
#define		k_NK											k_vii[42]
#define		k_count										k_vii[43]
#define		k_num_cells								k_vii[44]
#define		k_ngpu										k_vii[45]
#define		k_calc_area								k_vii[46]
#define		k_num_p2p									k_vii[47]
#define		k_num_part2								k_vii[48]
#define		k_penetration_solve				k_vii[49]
#define		k_multi_type						k_vii[50]
#define   k_noslip_bc								k_vii[51]
#define		k_IPF_kernel_type							k_vii[52]		// kernel type
#define		k_num_part3								k_vii[53]
#define		k_num_marker							k_vii[54]
#define   k_aps_solv                k_vii[55]
#define   k_structure_solv                k_vii[56]
#define 	k_h_change								k_vii[57]
#define  	k_viscosity_type							k_vii[58]
#define		k_timematchingstep										k_vii[59]		// time-step(s)

// Real type variable define
#define		k_kappa										k_vif[0]		// k in k*h
#define		k_p_ref										k_vif[1]		// reference pressure(for EOS)
#define		k_gamma										k_vif[2]		// gamma
#define		k_dt											k_vif[3]		// time-step(s)
#define		k_time										k_vif[4]		// time(s)
#define		k_time_end								k_vif[5]
// margin for simulation range
#define		k_Xmargin_m								k_vif[6]
#define		k_Xmargin_p								k_vif[7]
#define		k_Ymargin_m								k_vif[8]
#define		k_Ymargin_p								k_vif[9]
#define		k_Zmargin_m								k_vif[10]
#define		k_Zmargin_p								k_vif[11]
#define		k_u_limit									k_vif[12]
#define		k_c_xsph									k_vif[13]		// coefficient for XSPH
#define		k_c_repulsive							k_vif[14]		// coefficient for repulsive boundary force
//
//psh:: ISPH input
#define		k_drho_th									k_vif[15]		// density convergence criterion
#define		k_dp_th										k_vif[16]		// pressure convergence criterion
#define		k_p_relaxation						k_vif[17]		// relaxation factor for PCISPH pressure
//
//solution variables
#define		k_x_min										k_vif[18]
#define		k_x_max										k_vif[19]
#define		k_y_min										k_vif[20]
#define		k_y_max										k_vif[21]
#define		k_z_min										k_vif[22]
#define		k_z_max										k_vif[23]
#define		k_h_max									k_vif[24]
#define		k_soundspeed							k_vif[25]
#define		k_rho0_eos								k_vif[26]
//
#define		k_search_incr_factor			k_vif[27]
#define		k_search_kappa						k_vif[28]
#define		k_dcell										k_vif[29]
#define		k_ball_vel								k_vif[30]
#define		k_dt_structure											k_vif[31]		// time-step(s)
#define		k_time_output											k_vif[32]		// time-step(s)
#define		k_time2											k_vif[33]		// time-step(s)
#define		k_dt_2											k_vif[34]		// time-step(s)
#define		k_CFL											k_vif[35]		// time-step(s)
#define		k_CFL_ISPH											k_vif[36]		// time-step(s)

////////////////////////////////////////////////////////////////////////
void read_solv_input(int_t*vii,Real*vif,const char*FileName)
{
	solver_type=Wcsph;

	fp_solve=0;
	fv_solve=0;
	fg_solve=0;
	fs_solve=0;
	fb_solve=0;
	con_solve=0;
	boussinesq_solve=0;
	interface_solve=0;
	xsph_solve=0;
	kgc_solve=0;
	delSPH_solve=0;
	pst_solve=0;
	turbulence_model=0;
	concn_solve=0;
	c_xsph=0.0;
	c_repulsive=0.0;
	aps_solv=0;
	structure_solv=0;

	char inputString[1000];

	//inFile.open("../Result/output.txt");
	FILE*fd;
	fd=fopen(FileName,"r");

	int end;

	while(1){
		end=fscanf(fd,"%s",&inputString);		// reading one data from cc
		if(strcmp(inputString,"solver_type")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"WCSPH")==0) solver_type=Wcsph;
			if(strcmp(inputString,"ISPH")==0) solver_type=Isph;
			if(strcmp(inputString,"ICSPH")==0) solver_type=Icsph;

		}
		if(strcmp(inputString,"dimension(1/2/3):")==0){
			fscanf(fd,"%s",&inputString);
			dim=atoi(inputString);
		}
		if(strcmp(inputString,"property_table(YES/NO):")==0){
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"YES")==0) prop_table=1;
			if(strcmp(inputString,"NO")==0) prop_table=0;
		}
		if(strcmp(inputString,"kernel")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"Quartic")==0) kernel_type=Quartic;
			if(strcmp(inputString,"Gaussian")==0)kernel_type=Gaussian;
			if(strcmp(inputString,"Quintic")==0) kernel_type=Quintic;
			if(strcmp(inputString,"Wendland2")==0) kernel_type=Wendland2;
			if(strcmp(inputString,"Wendland4")==0) kernel_type=Wendland4;
			if(strcmp(inputString,"Wendland6")==0) kernel_type=Wendland6;
		}
		if(strcmp(inputString,"filter")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"Shepard")==0) flt_type=Shepard;
			if(strcmp(inputString,"MLS")==0) flt_type=MLS;
		}
		if(strcmp(inputString,"density")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"Direct")==0) rho_type=Mass_Sum;
			if(strcmp(inputString,"Continuity")==0) rho_type=Continuity;
		}
		if(strcmp(inputString,"time")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"Euler")==0) time_type=Euler;
			if(strcmp(inputString,"Predictor_Corrector")==0) time_type=Pre_Cor;
		}
		if(strcmp(inputString,"scheme")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"Eulerian")==0) scheme=Eulerian;
			if(strcmp(inputString,"Lagrangian")==0) scheme=Lagrangian;
			if(strcmp(inputString,"ALE")==0) scheme=ALE;
		}
		if(strcmp(inputString,"pressure-force")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"YES")==0) fp_solve=1;
			if(strcmp(inputString,"NO")==0) fp_solve=0;
		}
		if(strcmp(inputString,"viscous-force")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"NO")==0) fv_solve=0;
			if(strcmp(inputString,"Morris")==0) fv_solve=1;
			if(strcmp(inputString,"Monaghan")==0) fv_solve=2;
			if(strcmp(inputString,"Violeau")==0) fv_solve=3;
		}
		if(strcmp(inputString,"viscosity-model")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"Newtonian")==0) viscosity_type=0;
			if(strcmp(inputString,"Non-newtonian")==0) viscosity_type=1;
		}
		if(strcmp(inputString,"turbulence-model")==0){		//(by esk)
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"Laminar")==0) turbulence_model=0;
			if(strcmp(inputString,"k-lm")==0) turbulence_model=1;
			if(strcmp(inputString,"k-e")==0) turbulence_model=2;
			if(strcmp(inputString,"SPS")==0) turbulence_model=3;
			if(strcmp(inputString,"HB")==0) turbulence_model=4;
		}
		if(strcmp(inputString,"gravitational-force")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"YES")==0) fg_solve=1;
			if(strcmp(inputString,"NO")==0) fg_solve=0;
		}
		if(strcmp(inputString,"surface-tension-force")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"YES")==0) fs_solve=1;
			if(strcmp(inputString,"NO")==0) fs_solve=0;
		}
		if(strcmp(inputString,"surface-tension-model")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"Potential")==0) surf_model=1;
			if(strcmp(inputString,"Curvature")==0) surf_model=2;
		}
		if(strcmp(inputString,"IPF-kernel-type")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"Cosine")==0) IPF_kernel_type=Cosine;
			if(strcmp(inputString,"Gauss")==0)IPF_kernel_type=Gauss;
			if(strcmp(inputString,"Modified-Gaussian")==0) IPF_kernel_type=Modified_Gaussian;
			if(strcmp(inputString,"Cubic")==0) IPF_kernel_type=Cubic;
			if(strcmp(inputString,"Wend2")==0) IPF_kernel_type=Wend2;
		}
		if(strcmp(inputString,"interface-sharpness-force")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"YES")==0) interface_solve=1;
			if(strcmp(inputString,"NO")==0) interface_solve=0;
		}
		if(strcmp(inputString,"boundary-force")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"YES")==0) fb_solve=1;
			if(strcmp(inputString,"NO")==0) fb_solve=0;
		}
		if(strcmp(inputString,"Conduction(YES/NO):")==0){
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"YES")==0) con_solve=1;
			if(strcmp(inputString,"NO")==0) con_solve=0;
		}
		if(strcmp(inputString,"Boussinesq-natural-convection")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"YES")==0) boussinesq_solve=1;
			if(strcmp(inputString,"NO")==0) boussinesq_solve=0;
		}
		if(strcmp(inputString,"Concentration-diffusion(YES/NO):")==0){
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"YES")==0) concn_solve=1;
			if(strcmp(inputString,"NO")==0) concn_solve=0;
		}
		if(strcmp(inputString,"XSPH")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"YES")==0) xsph_solve=1;
			if(strcmp(inputString,"NO")==0) xsph_solve=0;
		}
		if(strcmp(inputString,"kernel-gradient-correction")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"NO")==0) kgc_solve=0;
			if(strcmp(inputString,"KGC")==0) kgc_solve=1;
			if(strcmp(inputString,"FPM")==0) kgc_solve=2;
			if(strcmp(inputString,"DFPM")==0) kgc_solve=3;
			if(strcmp(inputString,"KGF")==0) kgc_solve=4;
		}
		if(strcmp(inputString,"delta-SPH")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"NO")==0) delSPH_solve=0;
			if(strcmp(inputString,"Molteni")==0) delSPH_solve=1;
			if(strcmp(inputString,"Antuono")==0) delSPH_solve=2;
		}
		if(strcmp(inputString,"delta-SPH-model")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"Molteni")==0) delSPH_model=1;
			if(strcmp(inputString,"Antuono")==0) delSPH_model=2;
		}
		if(strcmp(inputString,"particle-shifting")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"YES")==0) pst_solve=1;
			if(strcmp(inputString,"NO")==0) pst_solve=0;
		}
		if(strcmp(inputString,"penetration_box")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"YES")==0) penetration_solve=1;
			if(strcmp(inputString,"NO")==0) penetration_solve=0;
		}
		if(strcmp(inputString,"noslip_boundary")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"YES")==0) noslip_bc=1;
			if(strcmp(inputString,"NO")==0) noslip_bc=0;
		}
		if(strcmp(inputString,"APS")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"YES")==0) aps_solv=1;
			if(strcmp(inputString,"NO")==0) aps_solv=0;
		}
		if(strcmp(inputString,"Variable")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"Abrupt")==0) h_change=1;
			if(strcmp(inputString,"Smooth")==0) h_change=2;
			if(strcmp(inputString,"NO")==0) h_change=0;
		}
		if(strcmp(inputString,"structure")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"YES")==0) structure_solv=1;
			if(strcmp(inputString,"NO")==0) structure_solv=0;
		}
		if(strcmp(inputString,"open")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"OPEN")==0) open_boundary=1;
			if(strcmp(inputString,"PERIODIC")==0) open_boundary=2;
			if(strcmp(inputString,"NO")==0) open_boundary=0;
		}
		if(strcmp(inputString,"Multi-resolution")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"sameh")==0) multi_type=0;
			if(strcmp(inputString,"stiff")==0) multi_type=1;
			if(strcmp(inputString,"smooth")==0) multi_type=2;
			if(strcmp(inputString,"stiff_conserve")==0) multi_type=3;
			if(strcmp(inputString,"smooth_conserve")==0) multi_type=4;
		}
		if(strcmp(inputString,"reference-pressure")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			p_ref=atof(inputString);
		}
		if(strcmp(inputString,"sound-speed:")==0){
			fscanf(fd,"%s",&inputString);
			soundspeed=atof(inputString);
		}
		if(strcmp(inputString,"reference-density-eos:")==0){
			fscanf(fd,"%s",&inputString);
			rho0_eos=atof(inputString);
		}
		if(strcmp(inputString,"gamma")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			gamma=atof(inputString);
		}
		if(strcmp(inputString,"kappa:")==0){
			fscanf(fd,"%s",&inputString);
			kappa=atof(inputString);
		}
		if(strcmp(inputString,"XSPH-coefficient:")==0){
			fscanf(fd,"%s",&inputString);
			c_xsph=atof(inputString);
		}
		if(strcmp(inputString,"Boundary-coefficient:")==0){
			fscanf(fd,"%s",&inputString);
			c_repulsive=atof(inputString);
		}
		if(strcmp(inputString,"minimum-iteration:")==0){
			fscanf(fd,"%s",&inputString);
			minIteration=atoi(inputString);
		}
		if(strcmp(inputString,"maximum-iteration:")==0){
			fscanf(fd,"%s",&inputString);
			maxIteration=atoi(inputString);
		}
		if(strcmp(inputString,"pressure-convergence")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			dp_th=atof(inputString);
		}
		if(strcmp(inputString,"density-convergence")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			drho_th=atof(inputString);
		}
		if(strcmp(inputString,"pressure-relaxation")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			p_relaxation=atof(inputString);
		}
		if(strcmp(inputString,"CFL_number")==0){
			fscanf(fd,"%s",&inputString);
			CFL=atof(inputString);
		}
		if(strcmp(inputString,"CFL_number_ISPH")==0){
			fscanf(fd,"%s",&inputString);
			CFL_ISPH=atof(inputString);
		}
		if(strcmp(inputString,"time-step")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			dt=atof(inputString);
		}
		if(strcmp(inputString,"time-step_ISPH")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			dt_2=atof(inputString);
		}
		if(strcmp(inputString,"start-time")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			time=atof(inputString);
		}
		if(strcmp(inputString,"end-time")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			time_end=atof(inputString);
		}
		if(strcmp(inputString,"ball-velocity")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			ball_vel=atof(inputString);
		}
		if(strcmp(inputString,"cell-initialization")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			freq_cell=atoi(inputString);
		}
		if(strcmp(inputString,"z-indexing")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"YES")==0) flag_z_index=1;
			if(strcmp(inputString,"NO")==0) flag_z_index=0;
		}
		if(strcmp(inputString,"neighbor")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"3X3")==0) nb_cell_type=0;
			if(strcmp(inputString,"5X5")==0) nb_cell_type=1;
		}
		if(strcmp(inputString,"timestep")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"YES")==0) flag_timestep_update=1;
			if(strcmp(inputString,"NO")==0) flag_timestep_update=0;
		}
		if(strcmp(inputString,"filtering")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			freq_filt=atoi(inputString);
		}
		if(strcmp(inputString,"density-renormalization")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			freq_mass_sum=atoi(inputString);
		}
		if(strcmp(inputString,"temperature-filtering")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			freq_temp=atoi(inputString);
		}
		if(strcmp(inputString,"freq-output")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			freq_output=atoi(inputString);
		}
		if(strcmp(inputString,"time-output")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			time_output=atof(inputString);
		}
		if(strcmp(inputString,"plot-variables:")==0){
			fscanf(fd,"%s",&inputString);
			num_plot_data=atoi(inputString);
			for(int ccount=0;ccount<num_plot_data;ccount++) {
				fscanf(fd,"%s",plot_data[ccount]);
			}
		}
		if(strcmp(inputString,"velocity-limit")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			u_limit=atof(inputString);
		}
		if(strcmp(inputString,"Xmargin(-):")==0){
			fscanf(fd,"%s",&inputString);
			Xmargin_m=atof(inputString);
		}
		if(strcmp(inputString,"Xmargin(+):")==0){
			fscanf(fd,"%s",&inputString);
			Xmargin_p=atof(inputString);
		}
		if(strcmp(inputString,"Ymargin(-):")==0){
			fscanf(fd,"%s",&inputString);
			Ymargin_m=atof(inputString);
		}
		if(strcmp(inputString,"Ymargin(+):")==0){
			fscanf(fd,"%s",&inputString);
			Ymargin_p=atof(inputString);
		}
		if(strcmp(inputString,"Zmargin(-):")==0){
			fscanf(fd,"%s",&inputString);
			Zmargin_m=atof(inputString);
		}
		if(strcmp(inputString,"Zmargin(+):")==0){
			fscanf(fd,"%s",&inputString);
			Zmargin_p=atof(inputString);
		}
		if(end==-1) break;
	}
	fclose(fd);
}
////////////////////////////////////////////////////////////////////////
// function calculating number of particles from input file
int_t gpu_count_particle_numbers(const char*FileName)
{
	int_t idx=0;
	int_t tmp,end;

	FILE*inFile;
	inFile=fopen(FileName,"r");

	while(1){
		end=fscanf(inFile,"%d\n",&tmp);
		if(end==-1) break;
		idx+=1;
	}
	fclose(inFile);
	return idx;
}
////////////////////////////////////////////////////////////////////////
// function calculating number of particles from input file
int_t gpu_count_particle_numbers2(const char*FileName)
{
	int_t idx=0;
	int_t nop;
	char buffer[1024];

	FILE*inFile;

	inFile=fopen(FileName,"r");
	while(fgets(buffer,1024,inFile)!=NULL) idx+=1;
	fclose(inFile);

	nop=idx-1;

	return nop;
}
////////////////////////////////////////////////////////////////////////
// function calculating number of boundary particles from input file
int_t gpu_count_boundary_numbers(const char*FileName)
{
	FILE*fd;
	fd=fopen("./input/p_type.txt","r");
	int_t end;
	int_t nb=0;		// number of boundary particles

	int tmp;

	// calculation of number of boundary particles
	while(1){
		end=fscanf(fd,"%d\n",&tmp);
		if(end==-1) break;
		else if(tmp==0) nb+=1;
	}
	nb=nb;				// actual number of bondary particles

	return nb;
}
/////////////////////////////////////////////////////////////
#define inp_x   1
#define inp_y   2
#define inp_z   3
#define inp_ux  4
#define inp_uy  5
#define inp_uz  6
#define inp_m   7
#define inp_ptype 8
#define inp_h 9
#define inp_temp  10
#define inp_pres  11
#define inp_rho 12
#define inp_rhoref  13
#define inp_ftotal  14
#define inp_concn 15
#define inp_cc 16
#define inp_vist  17
#define inp_ct_boundary 18
#define inp_hf_boundary 19
#define inp_lbl_surf  20
#define inp_drho  21
#define inp_denthalpy 22
#define inp_dconcn  23
#define inp_dk  24
#define inp_de  25
#define inp_ccc  26
// #define inp_eli			27
///////////////////////////////////////////////////////////////////////
void read_input(part1*Pa1)
{
  char FileName[256];
	strcpy(FileName,"./input/input.txt");

  char buffer[1024];
  char *tok;    //token of string

  int j,end,tmp,nov,nop;   //number of data, number of variables, number of partices
  int lbl_var[100];
  nov=nop=0;

  FILE*inFile;
  inFile=fopen(FileName,"r");

  // count number of variables
  fgets(buffer,1024-1,inFile);				// read first line
  tok=strtok(buffer,"\t");						// line segmentation
  while(tok!=NULL){
	  tmp=atoi(tok);
	  lbl_var[nov]=tmp;
	  nov++;														// count number of segments(variables)
	  tok=strtok(NULL,"\t");
  }

  // read data
  while(1){
		for(j=0;j<nov;j++){
			end=fscanf(inFile,"%s\n",buffer);
			if(end==-1) break;
			switch(lbl_var[j]){
			case inp_x:
				Pa1[nop].x=atof(buffer);
				// Pa1[nop].sx0=atof(buffer);
				break;
			case inp_y:
				Pa1[nop].y=atof(buffer);
				// Pa1[nop].sy0=atof(buffer);
				break;
			case inp_z:
				Pa1[nop].z=atof(buffer);
				// Pa1[nop].sz0=atof(buffer);
				break;
			case inp_ux:
				Pa1[nop].ux=atof(buffer);
				//Pa2[nop].ux0=atof(buffer);
				break;
			case inp_uy:
				Pa1[nop].uy=atof(buffer);
				//Pa2[nop].uy0=atof(buffer);
				break;
			case inp_uz:
				Pa1[nop].uz=atof(buffer);
				//Pa2[nop].uz0=atof(buffer);
				break;
			case inp_m:
				Pa1[nop].m=atof(buffer);
				break;
			case inp_ptype:
				Pa1[nop].p_type=atoi(buffer);
				break;
			case inp_h:
				Pa1[nop].h=atof(buffer);
				break;
			case inp_temp:
				Pa1[nop].temp=atof(buffer);
				Pa1[nop].temp0=atof(buffer);
				break;
			case inp_pres:
				Pa1[nop].pres=atof(buffer);
				break;
			case inp_rho:
				Pa1[nop].rho=atof(buffer);
				//Pa2[nop].rho0=atof(buffer);
				break;
			case inp_ftotal:
				//Pa3[nop].ftotal=atof(buffer);
				break;
			case inp_concn:
				Pa1[nop].concn=atof(buffer);
				//Pa2[nop].concn0=atof(buffer);
				break;
			case inp_vist:
				//Pa3[nop].vis_t=atof(buffer);
				break;
			case inp_ct_boundary:
				Pa1[nop].ct_boundary=atoi(buffer);
				break;
			case inp_lbl_surf:
				break;
			case inp_drho:
				//Pa3[nop].drho=atof(buffer);
				break;
			case inp_denthalpy:
				//Pa3[nop].denthalpy=atof(buffer);
				break;
			case inp_dconcn:
				//Pa3[nop].dconcn=atof(buffer);
				break;
			case inp_dk:
				//Pa3[nop].dk_turb=atof(buffer);
				break;
			case inp_de:
				//Pa3[nop].de_turb=atof(buffer);
				break;
			// case inp_ccc:
			// 	// Pa1[nop].pos=atoi(buffer);
			// 	Pa1[nop].ccc=atoi(buffer);
			// 	break;
				// case inp_eli:
				// 	Pa1[nop].eli=atof(buffer);
				//  break;
			default:
				printf("undefined variable name");
				break;
			}
		}
		if(end==-1) break;
		Pa1[nop].i_type=1;
		nop++;
  }
  fclose(inFile);
  printf("Input Files have been sucessfully read!!\n");
}
////////////////////////////////////////////////////////////////////////
void find_minmax(int_t*vii,Real*vif,part1*Pa1)
{
	int_t i;
	int_t end=num_part-1;

	Real min_x=Pa1[0].x;	Real max_x=Pa1[0].x;
	Real min_y=Pa1[0].y;	Real max_y=Pa1[0].y;
	Real min_z=Pa1[0].z;	Real max_z=Pa1[0].z;
	Real min_h=Pa1[0].h;	Real max_h=Pa1[0].h;
	//
	Real tmp_x,tmp_y,tmp_z,tmp_h;

	for(i=0;i<end;i++){
		tmp_x=Pa1[i].x;
		tmp_y=Pa1[i].y;
		tmp_z=Pa1[i].z;
		tmp_h=Pa1[i].h;

		if(tmp_x<min_x) min_x=tmp_x;
		if(tmp_x>max_x) max_x=tmp_x;
		if(tmp_y<min_y) min_y=tmp_y;
		if(tmp_y>max_y) max_y=tmp_y;
		if(tmp_z<min_z) min_z=tmp_z;
		if(tmp_z>max_z) max_z=tmp_z;
		if(tmp_h<min_h) min_h=tmp_h;
		if(tmp_h>max_h) max_h=tmp_h;
	}

	min_x-=(max_x-min_x)*Xmargin_m;
	max_x+=(max_x-min_x)*Xmargin_p;
	min_y-=(max_y-min_y)*Ymargin_m;
	max_y+=(max_y-min_y)*Ymargin_p;
	min_z-=(max_z-min_z)*Zmargin_m;
	max_z+=(max_z-min_z)*Zmargin_p;

	x_min=min_x;
	x_max=max_x;
	y_min=min_y;
	y_max=max_y;
	z_min=min_z;
	z_max=max_z;
	h_max=max_h;
}


void test_array(Real* datax, int size)
{
	printf("Test Array ---------------- \n");

	for(int i=0;i<size;i++)
	{
		printf("datax[%d]=%f\n",i,datax[i]);
	}

	printf("/n");

}

void read_table(const char*FileName){
	char inputString[1000];
	float data;
	char ch[100];
	char *ptr;

	int data_numbers=0;
	int count; count=0;
	int count2;
	int number_of_tables=10;
	int N_data[100];

	//inFile.open("../Result/output.txt");
	FILE*fd;
	fd=fopen(FileName,"r");

	while(1){
		// 한줄 읽기
		if(fgets(inputString,sizeof(inputString),fd)==NULL) break;
		// 첫마디 읽기
		ptr=strtok(inputString, " ");
		if(!strncmp(ptr,"#p_type",7)) {
		 	ptr=strtok(NULL," ");
			int number=atoi(ptr);

			host_table_index[number]=data_numbers;
			// number_of_tables++;

			while(1){
				fgets(inputString,sizeof(inputString),fd);

				if (!strncmp(inputString,"#end",4)) break;

				ptr=strtok(inputString, " ");
				if (!strncmp(ptr,"//T",3)) {
					fgets(inputString,sizeof(inputString),fd);
					ptr=strtok(inputString, ",");
					Real data=atof(ptr);
					host_Tab_T[data_numbers+count]=data;

					count++;
					while(ptr!=NULL){
						ptr=strtok(NULL, ",");
						if(ptr==NULL) break;
						data=atof(ptr);
						host_Tab_T[data_numbers+count]=data;
						count++;
					}
					count2=count;
					count=0;
				}
				else if (!strncmp(ptr,"//h",3)) {
					fgets(inputString,sizeof(inputString),fd);
					ptr=strtok(inputString, ",");
					Real data=atof(ptr);

					host_Tab_h[data_numbers+count]=data;

					count++;
					while(ptr!=NULL){
						ptr=strtok(NULL, ",");
						if(ptr==NULL) break;
						data=atof(ptr);
						host_Tab_h[data_numbers+count]=data;
						count++;
					}
					count2=count;
					count=0;
				}
				else if (!strncmp(ptr,"//k",3)) {
					fgets(inputString,sizeof(inputString),fd);
					ptr=strtok(inputString, ",");
					Real data=atof(ptr);

					host_Tab_k[data_numbers+count]=data;

					count++;
					while(ptr!=NULL){
						ptr=strtok(NULL, ",");
						if(ptr==NULL) break;
						data=atof(ptr);
						host_Tab_k[data_numbers+count]=data;
						count++;
					}
					count2=count;
					count=0;
				}
				else	if (!strncmp(ptr,"//cp",3)) {
					//printf("OK!\n");
					fgets(inputString,sizeof(inputString),fd);
					ptr=strtok(inputString, ",");
					Real data=atof(ptr);
					//printf("%f\n",data);
					host_Tab_cp[data_numbers+count]=data;
					// printf("data[%d]=%f\n",data_numbers+count,data);
					count++;
					while(ptr!=NULL)
					{
						ptr=strtok(NULL, ",");
						if(ptr==NULL) break;
						data=atof(ptr);
						host_Tab_cp[data_numbers+count]=data;
						// printf("data[%d]=%f\n",data_numbers+count,data);
						count++;
					}
					count2=count;
					count=0;
					//fgets(inputString,sizeof(inputString),fd);
					//ptr=inputString;
				}
				else if (!strncmp(ptr,"//mu",3)) {
					fgets(inputString,sizeof(inputString),fd);
					ptr=strtok(inputString, ",");
					Real data=atof(ptr);

					host_Tab_vis[data_numbers+count]=data;

					count++;
					while(ptr!=NULL){
						ptr=strtok(NULL, ",");
						if(ptr==NULL) break;
						data=atof(ptr);
						host_Tab_vis[data_numbers+count]=data;

						count++;
					}
					count2=count;
					count=0;
				}
			}
			data_numbers=data_numbers+count2;
			host_table_size[number]=count2;
			// N_data[number]=data_numbers;
		}
	}
	fclose(fd);

	// 추가계산
	// int Table_size[10];
	// int Table_index[10];
	// host_table_index[0]=0;
	// host_table_size[0]=N_data[0];
	// for (int i=1;i<number_of_tables;i++){
	// 	host_table_index[i]=N_data[i-1];
	// 	host_table_size[i]=N_data[i]-N_data[i-1];
	// }

	// print for check
	printf("\nsaved results-----------\n\n");
	printf("number of tables=%d\n\n",number_of_tables);
	printf("data numbers=%d\n\n",data_numbers);
	for (int i=0;i<data_numbers;i++){
		printf("T[%d]=%f\n",i,host_Tab_T[i]);
	}
	printf("\n");
	for (int i=0;i<data_numbers;i++){
		printf("h[%d]=%f\n",i,host_Tab_h[i]);
	}
	printf("\n");
	for (int i=0;i<data_numbers;i++){
		printf("k[%d]=%f\n",i,host_Tab_k[i]);
	}
		printf("\n");
	for (int i=0;i<data_numbers;i++)
	{
		printf("cp[%d]=%f\n",i,host_Tab_cp[i]);
	}
	printf("\n");
	for (int i=0;i<data_numbers;i++){
		printf("mu[%d]=%f\n",i,host_Tab_vis[i]);
	}
	// printf("\n");
	// for (int i=0;i<number_of_tables;i++){
	// 	printf("N_data[%d]=%d\n",i,N_data[i]);
	// }
	printf("\n");
	for (int i=0;i<number_of_tables;i++){
		printf("Table_size[%d]=%d\n",i,host_table_size[i]);
	}
	printf("\n");
	for (int i=0;i<number_of_tables;i++){
		printf("Table_index[%d]=%d\n",i,host_table_index[i]);
	}
	printf("\n");
}
