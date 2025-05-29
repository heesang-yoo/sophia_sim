// Cparticle Class Declaration
// Cparticle class contains particle information.
#ifndef max
#define max(a,b) (((a)>(b))?(a):(b))
#endif

#ifndef min
#define min(a,b) (((a)<(b))?(a):(b))
#endif
////////////////////////////////////////////////////////////////////////
typedef struct particles_array_1{
	uint_t i_type;													// Inner or Outer
	uint_t buffer_type;											// buffer_type (0: active , 1: inlet, 2: outlet)
	uint_t p_type;													// particle type: FLUID or BOUNDARY
	uint_t ct_boundary;												// const temperatrue particle index (const temp particle :1 / else : 0)

	Real x,y,z;															// (Predicted) positions [m] ( Predictor_Corrector : Predicted position / Euler : Real Position )
	Real x_star,y_star,z_star;
	Real ux,uy,uz;													// (Predicted) velocity [m/s] ( Predictor_Corrector : Predicted velocity / Euler : Real Velocity )
	Real m,m0;																	// mass [kg]
	Real h,h0;																	// kernel distance
	Real temp;															// temperature [K]
	Real pres, dpres;															// pressure [Pa]
	//Real pres_ave;													// averaged pressure
	Real rho;																// density [kg/m3]	( Predictor_Corrector : Predicted density / Euler : Real density )
	Real temp0;

	Real flt_s;															// Shepard filter
	Real w_dx;															// w(dx) for particle shifting
	Real enthalpy;
	Real concn;
	Real shiftx, shifty;
	
	Real grad_rhox,grad_rhoy,grad_rhoz;				// density gradient - only mass kernel using

	// turbulence (by esk)
	Real k_turb,e_turb;												// turbulence kinetic energy,dissipation rate --> check unit

	// ALE (by hsy)
 	Real elix, eliy, eliz;
	Real vortx, vorty, vortz;
	Real concentration;
	Real vol0, vol;

	// // // for Adaptive Particle Definement (DHK)
	// int_t aps_fine;
	// int_t aps_cond;
	// int_t split_num;
	// int_t merge_num;
	// int_t merge_flag;
	// int_t isolation_count;
	// int_t pressure_flag;

 	// Real cell_prop;
	// int_t cell_num;

	// // for Immersed Boundary Method (IBM)

	Real ux_i, uy_i, uz_i;
	Real ax, ay, az;
	Real fbx, fby, fbz;
	Real fcx, fcy, fcz;
	int_t pos;
	Real PPE1, PPE2;

	// // // Structure modeling (LTH)
	// int_t ccc;								// Label
	// Real F11,F12,F13,F21,F22,F23,F31,F32,F33;	// Deformation tensor
	// Real R11,R12,R13,R21,R22,R23,R31,R32,R33;	// Rotational tentor
	// Real q0,q1,q2,q3;
	// Real p11,p12,p13,p21,p22,p23,p31,p32,p33;	// Piola-Kirchhoff stress tensor
	// Real px,py,pz;								// Magnitude of stress tensor
	// Real jacob;									// Jacobian of Deformation tensor
	// Real sx0, sy0, sz0;
		
	int_t ncell;

	Real dux, duy;
	Real D,newvis;
	Real uanalytic;
	
}part1;
////////////////////////////////////////////////////////////////////////
typedef struct particles_array_2{
	Real rho_ref;

	//// turbulence (by esk)
	Real SR;																	// strain rate (2S:S)

	Real x0,y0,z0;													// Initial positions [m]
	Real ux0,uy0,uz0;												// Initial velocity [m/s]
	Real rho0;															// Initial density [kg/m3]
	Real drho0;														// Error compensation: divergence error
	Real vol0;
	Real dvol0;
	Real temp0;

	// psh: concentration diffusion
	Real concn0;														// concentration
	Real enthalpy0;													// enthalpy [J/kg]
}part2;
////////////////////////////////////////////////////////////////////////
typedef struct particles_array_3{
	Real drho;															// Time Derivative of density [kg/m3 s]
	Real dvol;
	Real dconcn;														// concentration time derivative
	Real denthalpy;
	Real ftotalx,ftotaly,ftotalz;						// total force [m/s2]
	Real fpx,fpy,fpz;
	Real ftotal;
	Real dtemp;

	// turbulence (by esk)
	Real vis_t;																// turbulence viscosity
	Real Sxx,Sxy,Sxz,Syy,Syz,Szz;							// strain tensor... for SPH model
	Real dk_turb,de_turb;											// turbulence kinetic energy,dissipation rate --> check unit

	Real lbl_surf;													// surface label
	Real cc;																// color code

	Real Cm[Correction_Matrix_Size][Correction_Matrix_Size];
	Real A[Correction_Matrix_Size][Correction_Matrix_Size];
	// Real inv_cm_xx,inv_cm_yy,inv_cm_zz;
	// Real inv_cm_xy,inv_cm_yz,inv_cm_zx;

	Real lambda;
	Real nx,ny,nz;														// color code gradient for surface tension force (2016.09.02 jyb)
	Real nx_c,ny_c,nz_c;											// color code gradient for surface tension force (2016.09.02 jyb)
	Real nx_s,ny_s,nz_s;
	Real tx_s,ty_s,tz_s;
	Real nx_w,ny_w,nz_w;
	Real color;
	
	Real nmag;																// 2017.04.20 jyb
	Real nmag_c;															// 2017.04.20 jyb
	Real curv;
	Real fsx,fsy,fsz;
		Real vort_x,vort_y,vort_z;
	Real Gxx, Gxy, Gyx, Gyy;
	Real L11, L12, L21, L22;
	Real A111, A112, A122, A211, A212, A222;

	Real fbdx, fbdy, fbdz;


}part3;
///////////////////////////////////////////////////////////////////////////
typedef struct p2p_particles_array_3{
	Real drho;															// Time Derivative of density [kg/m3 s]
	Real dconcn;														// concentration time derivative
	Real denthalpy;
	Real ftotalx,ftotaly,ftotalz;						// total force [m/s2]
	Real ftotal;
}p2p_part3;
///////////////////////////////////////////////////////////////////////////
typedef struct mesh_array{
	Real xinit, yinit, zinit;
	Real uavg, vavg, wavg;
	Real pres;
	int_t idx;
}mesh;
////////////////////////////////////////////////////////////////////////
