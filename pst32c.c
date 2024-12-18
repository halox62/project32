/**************************************************************************************
* 
* CdL Magistrale in Ingegneria Informatica
* Corso di Architetture e Programmazione dei Sistemi di Elaborazione - a.a. 2024/25
* 
* Progetto dell'algoritmo Predizione struttura terziaria proteine 221 231 a
* in linguaggio assembly x86-32 + SSE
* 
* F. Angiulli F. Fassetti S. Nisticò, novembre 2024
* 
**************************************************************************************/

/*
* 
* Software necessario per l'esecuzione:
* 
*    NASM (www.nasm.us)
*    GCC (gcc.gnu.org)
* 
* entrambi sono disponibili come pacchetti software 
* installabili mediante il packaging tool del sistema 
* operativo; per esempio, su Ubuntu, mediante i comandi:
* 
*    sudo apt-get install nasm
*    sudo apt-get install gcc
* 
* potrebbe essere necessario installare le seguenti librerie:
* 
*    sudo apt-get install lib32gcc-4.8-dev (o altra versione)
*    sudo apt-get install libc6-dev-i386
* 
* Per generare il file eseguibile:
* 
* nasm -f elf32 pst32.nasm && gcc -m32 -msse -O0 -no-pie sseutils32.o pst32.o pst32c.c -o pst32c -lm && ./pst32c $pars
* 
* oppure
* 
* ./runpst32
* 
*/


#include <iso646.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <libgen.h>
#include <xmmintrin.h>


//#define	type		float
#define type 		double 
#define	MATRIX		type*
#define	VECTOR		type*

#define FACT2 (2*1)     
#define FACT3 (3*2*1)   
#define FACT4 (4*3*2*1)
#define FACT5 (5*4*3*2*1)
#define FACT6 (6*5*4*3*2*1)
#define FACT7 (7*6*5*4*3*2*1)

#define random() (((type) rand())/RAND_MAX)

type hydrophobicity[] = {1.8, -1, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -1, -3.9, 3.8, 1.9, -3.5, -1, -1.6, -3.5, -4.5, -0.8, -0.7, -1, 4.2, -0.9, -1, -1.3, -1};		// hydrophobicity
type volume[] = {88.6, -1, 108.5, 111.1, 138.4, 189.9, 60.1, 153.2, 166.7, -1, 168.6, 166.7, 162.9, 114.1, -1, 112.7, 143.8, 173.4, 89.0, 116.1, -1, 140.0, 227.8, -1, 193.6, -1};		// volume
type charge[] = {0, -1, 0, -1, -1, 0, 0, 0.5, 0, -1, 1, 0, 0, 0, -1, 0, 0, 1, 0, 0, -1, 0, 0, -1, 0, -1};		// charge

typedef struct {
	char* seq;		// sequenza di amminoacidi
	int N;			// lunghezza sequenza
	unsigned int sd; 	// seed per la generazione casuale
	type to;		// temperatura INIZIALE
	type alpha;		// tasso di raffredamento
	type k;		// costante
	VECTOR hydrophobicity; // hydrophobicity
	VECTOR volume;		// volume
	VECTOR charge;		// charge
	VECTOR phi;		// vettore angoli phi
	VECTOR psi;		// vettore angoli psi
	type e;		// energy
	int display;
	int silent;

} params;


/*
* 
*	Le funzioni sono state scritte assumento che le matrici siano memorizzate 
* 	mediante un array (float*), in modo da occupare un unico blocco
* 	di memoria, ma a scelta del candidato possono essere 
* 	memorizzate mediante array di array (float**).
* 
* 	In entrambi i casi il candidato dovr� inoltre scegliere se memorizzare le
* 	matrici per righe (row-major order) o per colonne (column major-order).
*
* 	L'assunzione corrente � che le matrici siano in row-major order.
* 
*/

void* get_block(int size, int elements) { 
	return _mm_malloc(elements*size,16); 
}

void free_block(void* p) { 
	_mm_free(p);
}

MATRIX alloc_matrix(int rows, int cols) {
	return (MATRIX) get_block(sizeof(type),rows*cols);
}

int* alloc_int_matrix(int rows, int cols) {
	return (int*) get_block(sizeof(int),rows*cols);
}

char* alloc_char_matrix(int rows, int cols) {
	return (char*) get_block(sizeof(char),rows*cols);
}

void dealloc_matrix(void* mat) {
	free_block(mat);
}

/*
* 
* 	load_data
* 	=========
* 
*	Legge da file una matrice di N righe
* 	e M colonne e la memorizza in un array lineare in row-major order
* 
* 	Codifica del file:
* 	primi 4 byte: numero di righe (N) --> numero intero
* 	successivi 4 byte: numero di colonne (M) --> numero intero
* 	successivi N*M*4 byte: matrix data in row-major order --> numeri floating-point a precisione singola
* 
*****************************************************************************
*	Se lo si ritiene opportuno, � possibile cambiare la codifica in memoria
* 	della matrice. 
*****************************************************************************
* 
*/

MATRIX load_data(char* filename, int *n, int *k) {
	FILE* fp;
	int rows, cols, status, i;
	
	fp = fopen(filename, "rb");
	
	if (fp == NULL){
		printf("'%s': bad data file name!\n", filename);
		exit(0);
	}
	
	status = fread(&cols, sizeof(int), 1, fp);
	status = fread(&rows, sizeof(int), 1, fp);
	
	MATRIX data = alloc_matrix(rows,cols);
	status = fread(data, sizeof(type), rows*cols, fp);
	fclose(fp);
	
	*n = rows;
	*k = cols;
	
	return data;
}

/*
* 
* 	load_seq
* 	=========
* 
*	Legge da file una matrice di N righe
* 	e M colonne e la memorizza in un array lineare in row-major order
* 
* 	Codifica del file:
* 	primi 4 byte: numero di righe (N) --> numero intero
* 	successivi 4 byte: numero di colonne (M) --> numero intero
* 	successivi N*M*1 byte: matrix data in row-major order --> charatteri che compongono la stringa
* 
*****************************************************************************
*	Se lo si ritiene opportuno, � possibile cambiare la codifica in memoria
* 	della matrice. 
*****************************************************************************
* 
*/

char* load_seq(char* filename, int *n, int *k) {
	FILE* fp;
	int rows, cols, status, i;
	
	fp = fopen(filename, "rb");

	
	
	if (fp == NULL){
		printf("'%s': bad data file name!\n", filename);
		exit(0);
	}
	
	status = fread(&cols, sizeof(int), 1, fp);
	status = fread(&rows, sizeof(int), 1, fp);

	
	char* data = alloc_char_matrix(rows,cols);
	status = fread(data, sizeof(char), rows*cols, fp);
	fclose(fp);
	
	*n = rows;
	*k = cols;
	
	return data;
}

/*
* 	save_data
* 	=========
* 
*	Salva su file un array lineare in row-major order
*	come matrice di N righe e M colonne
* 
* 	Codifica del file:
* 	primi 4 byte: numero di righe (N) --> numero intero a 32 bit
* 	successivi 4 byte: numero di colonne (M) --> numero intero a 32 bit
* 	successivi N*M*4 byte: matrix data in row-major order --> numeri interi o floating-point a precisione singola
*/

void save_data(char* filename, void* X, int n, int k) {
	FILE* fp;
	int i;
	fp = fopen(filename, "wb");
	if(X != NULL){
		fwrite(&k, 4, 1, fp);
		fwrite(&n, 4, 1, fp);
		for (i = 0; i < n; i++) {
			fwrite(X, sizeof(type), k, fp);
			//printf("%i %i\n", ((int*)X)[0], ((int*)X)[1]);
			X += sizeof(type)*k;
		}
	}
	else{
		int x = 0;
		fwrite(&x, 4, 1, fp);
		fwrite(&x, 4, 1, fp);
	}
	fclose(fp);
}

/*
* 	save_out
* 	=========
* 
*	Salva su file un array lineare composto da k elementi.
* 
* 	Codifica del file:
* 	primi 4 byte: contenenti l'intero 1 		--> numero intero a 32 bit
* 	successivi 4 byte: numero di elementi k     --> numero intero a 32 bit
* 	successivi byte: elementi del vettore 		--> k numero floating-point a precisione singola
*/

void save_out(char* filename, MATRIX X, int k) {
	FILE* fp;
	int i;
	int n = 1;
	fp = fopen(filename, "wb");
	if(X != NULL){
		fwrite(&n, 4, 1, fp);
		fwrite(&k, 4, 1, fp);
		fwrite(X, sizeof(type), k, fp);
	}
	fclose(fp);
}

/*
* 	gen_rnd_mat
* 	=========
* 
*	Genera in maniera casuale numeri reali tra -pi e pi
*	per riempire una struttura dati di dimensione Nx1
* 
*/

void gen_rnd_mat(VECTOR v, int N){
	int i;

	for(i=0; i<N; i++){
		// Campionamento del valore + scalatura
		v[i] = (random()*2 * M_PI) - M_PI;
	}
}

// PROCEDURE ASSEMBLY
//extern void prova(params* input);


 
// Funzione per calcolare coseno con approssimazione
double cos_approx(double x) {
    double x2 = x * x; 
    return 1 - (x2 / FACT2) + (x2 * x2 / FACT4) - (x2 * x2 * x2 / FACT6);
}
 
// Funzione per calcolare seno con approssimazione
double sin_approx(double x) {
    double x2 = x * x; 
    return x - (x * x2 / FACT3) + (x * x2 * x2 / FACT5) - (x * x2 * x2 * x2 / FACT7);
}

void rotation(VECTOR axis, type theta, MATRIX R){

	double norm = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2];
	axis[0] /= norm;
	axis[1] /= norm;
	axis[2] /= norm;
    


	type a = cos_approx(theta / 2.0);
    type b = -axis[0] * sin_approx(theta / 2.0);
	type c = -axis[1] * sin_approx(theta / 2.0);
    type d = -axis[2] * sin_approx(theta / 2.0);


	type aa = a * a, bb = b * b, cc = c * c, dd = d * d;
	type bc = b * c, ad = a * d, cd = c * d, ab = a * b, bd = b * d, ac = a * c;

	R[0] = aa + bb - cc - dd;

	R[1] = 2.0 * (bc + ad);
	R[2] = 2.0 * (bd - ac);

	R[3] = 2.0 * (bc - ad);
	R[4] =  aa + cc - bb - dd;
	R[5] = 2.0 * (cd + ab);

	R[6] = 2.0 * (bd + ac);
	R[7] = 2.0 * (cd - ab);
	R[8] = aa + dd - bb - cc;

}

MATRIX matrixProd(MATRIX m1, MATRIX m2){
	MATRIX res = (MATRIX)aligned_alloc(16, 3 * sizeof(type));
	res[0]=m1[0]*m2[0]+m1[1]*m2[3]+m1[2]*m2[6];
	res[1]=m1[0]*m2[1]+m1[1]*m2[4]+m1[2]*m2[7];
	res[2]=m1[0]*m2[2]+m1[1]*m2[5]+m1[2]*m2[8];
	return res;
}


MATRIX backbone(int n, VECTOR phi, VECTOR psi){
	type* v1=alloc_matrix(3,1);
	type* v2=alloc_matrix(3,1);
	type* v3=alloc_matrix(3,1);
	type norma=0.0;
	MATRIX newv=(MATRIX)malloc(3*sizeof(type));
	MATRIX R = (MATRIX)malloc(9*sizeof(type));
	const type r_ca_n = 1.46;
	const type r_ca_c = 1.52;
	const type r_c_n = 1.33;

	const type theta_ca_n_c = 2.028;
	const type theta_c_n_ca = 2.124;
	const type theta_n_ca_c = 1.940;

	MATRIX coords = (MATRIX)aligned_alloc(16, ((n * 3) * 3) * sizeof(type));
	MATRIX prod = (MATRIX)aligned_alloc(16, 3 * sizeof(type));
	
	coords[0]=0.0;
	coords[1]=0.0;
	coords[2]=0.0;//N

	coords[3]=r_ca_n;
	coords[4]=0.0;
	coords[5]=0.0;//Ca


	//	N	 Ca	   C
	//0,0,0,1,0,0,1,0,0

	for(int i=0; i<n ;i++){
		int idx= i*9;
		if(i>0){
			//Posiziona N usando l'ultimo C
			v1[0]=coords[idx-3]-coords[idx-6];
			v1[1]=coords[idx-2]-coords[idx-5];
			v1[2]=coords[idx-1]-coords[idx-4];


			norma=sqrt((v1[0]*v1[0])+(v1[1]*v1[1])+(v1[2]*v1[2]));

			v1[0]=v1[0] / norma;
			v1[1]=v1[1] / norma;
			v1[2]=v1[2] / norma;

			newv[0]=0.0;
			newv[1]=r_c_n;
			newv[2]=0.0;

			rotation(v1, theta_c_n_ca, R);
			
			prod=matrixProd(newv,R);

			coords[idx]=coords[idx-3]+prod[0];
			coords[idx+1]=coords[idx-2]+prod[1];
			coords[idx+2]=coords[idx-1]+prod[2];


			//Posiziona Ca usando l'ultimo phi
			v2[0]=coords[idx]-coords[idx-3];
			v2[1]=coords[idx+1]-coords[idx-2];
			v2[2]=coords[idx+2]-coords[idx-1];

			norma=sqrt(v2[0]*v2[0]+v2[1]*v2[1]+v2[2]*v2[2]);

			v2[0]=v2[0] / norma;
			v2[1]=v2[1] / norma;
			v2[2]=v2[2] / norma;

			newv[0]=0.0;
			newv[1]=r_ca_n;
			newv[2]=0.0;

			rotation(v2, phi[i], R);

			prod=matrixProd(newv,R);
			//prod=matrixVectorMultiplication(R, newv);

			coords[idx+3]=coords[idx]+prod[0];
			coords[idx+4]=coords[idx+1]+prod[1];
			coords[idx+5]=coords[idx+2]+prod[2];
		}
		//Posiziona C usando l'ultimo psi
		v3[0]=coords[idx+3]-coords[idx];
		v3[1]=coords[idx+4]-coords[idx+1];
		v3[2]=coords[idx+5]-coords[idx+2];

		norma=sqrt(v3[0]*v3[0]+v3[1]*v3[1]+v3[2]*v3[2]);


		v3[0]=v3[0] / norma;
		v3[1]=v3[1] / norma;
		v3[2]=v3[2] / norma;

		newv[0]=0.0;
		newv[1]=r_ca_c;
		newv[2]=0.0;

		rotation(v3, psi[i], R);

		/*for(int i=0;i<9;i++){
			printf("R: %f\n",R[i]);
		}*/

		//printf("psi:%f\n",psi[i]);

		prod=matrixProd(newv,R);
		//prod=matrixVectorMultiplication(R, newv);

		/*for(int i=0;i<3;i++){
			printf("prod[%d]:%f\n",i,prod[i]);
		}*/

		coords[idx+6]=coords[idx+3]+prod[0];
		coords[idx+7]=coords[idx+4]+prod[1];
		coords[idx+8]=coords[idx+5]+prod[2];

	}
	return coords;
}

type rama_energy(int n, VECTOR phi, VECTOR psi){
  	type e=0.0;
	type alpha_phi = -57.8, alpha_psi = -47.0;
	type beta_phi = -119.0, beta_psi = 113.0;

	for(int i = 0; i<n; i++){
	    type alpha_dist = sqrt( (phi[i]-alpha_phi)*(phi[i]-alpha_phi) + (psi[i]-alpha_psi)*(psi[i]-alpha_psi) );
	    type beta_dist = sqrt( (phi[i]-beta_phi)*(phi[i]-beta_phi) + (psi[i]-beta_psi)*(psi[i]-beta_psi) );
	    e = e + 0.5*fmin(alpha_dist, beta_dist);
	}
	return e;
}

type min(type a, type b){
    if(a<b){
        return a;
    }
    return b;
}

MATRIX coordinate_c_alpha(MATRIX coords, int n){
	MATRIX ris = aligned_alloc(16, ((n*3))*sizeof(type));
	int idx = 0;
	for(int i = 3; i<n*9; i = i+9){
		ris[idx] = coords[i];
		ris[idx+1] = coords[i+1];
		ris[idx+2] = coords[i+2];
		idx+=3;
	}
	return ris;
}

//	N     Ca	C    N     Ca
//0,0,0,1,1,1,2,2,2,3,3,3,4,4,4

float dist(VECTOR v1, VECTOR v2){
	return sqrt((v2[0]-v1[0])*(v2[0]-v1[0]) + (v2[1]-v1[1])*(v2[1]-v1[1]) + (v2[2]-v1[2])*(v2[2]-v1[2]));
}


type hydrophobic_energy(int n, char* s, MATRIX coords){
	type e=0.0;
    MATRIX ca_coords = coordinate_c_alpha(coords, n);
    type* v1=alloc_matrix(3,1);
	type* v2=alloc_matrix(3,1);
	int ammI=0;
	int ammJ=0;
	int indxI=0;
	int indxJ=0;

	for(int i = 0; i<n; i++){
       	v1[0] = ca_coords[indxI];
        v1[1] = ca_coords[indxI+1];
        v1[2] = ca_coords[indxI+2]; //abbiamo le coordinate dell'i-esimo amminoacido in v1

		indxJ=i*3+3;

		for(int j = i+1; j<n;j++){
			v2[0] = ca_coords[indxJ];
			v2[1] = ca_coords[indxJ+1];
            v2[2] = ca_coords[indxJ+2]; //abbiamo le coordinate del j-esimo amminoacido in v2
			
			type distV=dist(v1,v2);
            if(distV>0 and distV<10.0){
				ammI=((int)s[i])-65;
				ammJ=((int)s[j])-65;
				e = e + ((hydrophobicity[ammI] * hydrophobicity[ammJ])/distV);
			}
            indxJ+=3;
        }
		indxI+=3;
    }
	return e;
}



type electrostatic_energy(int n, char* s, MATRIX coords){
	type e=0.0;
	MATRIX ca_coords = coordinate_c_alpha(coords, n);
	int ammI=0;
	int ammJ=0;
	int indxI=0;
	int indxJ=3;

	type v1[3], v2[3];

	for(int i = 0; i<n; i++){
		v1[0] = ca_coords[indxI];
		v1[1] = ca_coords[indxI+1];
		v1[2] = ca_coords[indxI+2];

		indxJ=i*3+3;

		for(int j = i+1; j<n; j++){

			v2[0] = ca_coords[indxJ];
			v2[1] = ca_coords[indxJ+1];
			v2[2] = ca_coords[indxJ+2];

			ammI=(int)s[i]-65;
			ammJ=(int)s[j]-65;

			type distV=dist(v1,v2);

			if(i!=j and distV>0 and distV<10.0 and charge[ammI]!=0 and charge[ammJ])
				e = e + (charge[ammI] * charge[ammJ])/(distV*4.0);

			indxJ+=3;
		}
		indxI+=3;

	}
	return e;
}
//0,0,0,1,1,1,2,2,2
//i=3
//j=6

type packing_energy(int n, char* s, MATRIX coords){
	MATRIX ca_coords = coordinate_c_alpha(coords, n);
	type e=0.0;
	int ammI=0;
	int ammJ=0;
	type v1[3], v2[3];
	int indxI=0;
	int indxJ=0;

	for(int i = 0; i<n; i++){

		v1[0] = ca_coords[indxI];
		v1[1] = ca_coords[indxI+1];
		v1[2] = ca_coords[indxI+2];    //abbiamo le coordinate dell'i-esimo amminoacido in v1

		type density=0;

		for(int j = 0; j<n; j++){

			v2[0] = ca_coords[indxJ];
			v2[1] = ca_coords[indxJ+1];
			v2[2] = ca_coords[indxJ+2];  //abbiamo le coordinate del j-esimo amminoacido in v2

			
			type distV=dist(v1, v2);
			if(i!=j and distV<10.0){
				ammI=((int)s[i])-65;
				ammJ=((int)s[j])-65;
				density = density + (volume[ammJ]/(distV*distV*distV));
			}
			indxJ+=3;
		}
		indxJ=0;
		indxI+=3;
		e = e + ((volume[ammI]-(density))*(volume[ammI]-(density)));
	}

	return e;
}


type energy(int n, char* s, VECTOR phi, VECTOR psi){
	MATRIX coords = backbone(n, phi, psi);
	type rama_e = rama_energy(n, phi, psi);
	//printf("rama_energy: %f\n",rama_e);
	type hydro_e = hydrophobic_energy(n, s,coords);
	//printf("hydro_energy: %f\n",hydro_e);
	type elec_e = electrostatic_energy(n, s, coords);
	//printf("elec_e: %f\n",elec_e);
	type pack_e = packing_energy(n, s, coords);
	//printf("pack_energy: %f\n",pack_e);
	// Pesi per i diversi contributi
	type w_rama = 1.0, w_hydro = 0.5, w_elec = 0.2, w_pack = 0.3;
	//Energia totale
	type total_e = w_rama*rama_e + w_hydro*hydro_e + w_elec*elec_e + w_pack*pack_e;
	/*printf("%f\n",total_e);
	for(int i=0;i<35;i++){
		printf("coords[%d]=%f\n",i,coords[i]);
	}
	exit(1);*/
	return total_e;
}





void pst(params* input){
	type energy_p=0.0;
	char* aminoacidi = input->seq;
	int n= input->N;
	VECTOR v_phi = input->phi;
	VECTOR v_psi = input->psi;
	type T=input->to;
	energy_p = energy(n,aminoacidi, v_phi, v_psi);
	printf("Value of energy_p: %f\n", energy_p);
	type t=0.0;
	do{
		int i = random()*n;
		type delta_phi =  (random()*2 * M_PI) - M_PI;
		v_phi[i] = v_phi[i] + delta_phi;
		type delta_psi =  (random()*2 * M_PI) - M_PI;
		v_psi[i] = v_psi[i] + delta_psi;
		type delta_e = energy(n,aminoacidi, v_phi, v_psi)-energy_p;
		if(delta_e<=0.0){
			energy_p = energy(n,aminoacidi, v_phi, v_psi);
		}else{
			type p=exp(-(delta_e)/(input->k*T));
			type r=random();
			if(r<=p){
				energy_p = energy(n, aminoacidi, v_phi, v_psi);
			}else{
				v_phi[i] = v_phi[i] - delta_phi;
				v_psi[i] = v_psi[i] - delta_psi;
			}
		}
		t=t+1;
		T=input->to-sqrt(input->alpha*t);
		}while(T>=0);

		printf("ENERGY:%f\n", energy_p);

		//free(v_phi);
		//free(v_psi);
}


int main(int argc, char** argv) {
	char fname_phi[256];
	
	char fname_psi[256];
	char* seqfilename = NULL;
	clock_t t;
	float time;
	int d;

	

	
	//
	// Imposta i valori di default dei parametri
	//
	params* input = malloc(sizeof(params));
	input->seq = NULL;	
	input->N = -1;			
	input->to = -1;
	input->alpha = -1;
	input->k = -1;		
	input->sd = -1;		
	input->phi = NULL;		
	input->psi = NULL;
	input->silent = 0;
	input->display = 0;
	input->e = -1;
	input->hydrophobicity = hydrophobicity;
	input->volume = volume;
	input->charge = charge;

	


	//
	// Visualizza la sintassi del passaggio dei parametri da riga comandi
	//
	if(argc <= 1){
		printf("%s -seq <SEQ> -to <to> -alpha <alpha> -k <k> -sd <sd> [-s] [-d]\n", argv[0]);
		printf("\nParameters:\n");
		printf("\tSEQ: il nome del file ds2 contenente la sequenza amminoacidica\n");
		printf("\tto: parametro di temperatura\n");
		printf("\talpha: tasso di raffredamento\n");
		printf("\tk: costante\n");
		printf("\tsd: seed per la generazione casuale\n");
		printf("\nOptions:\n");
		printf("\t-s: modo silenzioso, nessuna stampa, default 0 - false\n");
		printf("\t-d: stampa a video i risultati, default 0 - false\n");
		exit(0);
	}

	//
	// Legge i valori dei parametri da riga comandi
	//

	

	int par = 1;
	while (par < argc) {
		if (strcmp(argv[par],"-s") == 0) {
			input->silent = 1;
			par++;
		} else if (strcmp(argv[par],"-d") == 0) {
			input->display = 1;
			par++;
		} else if (strcmp(argv[par],"-seq") == 0) {
			par++;
			if (par >= argc) {
				printf("Missing dataset file name!\n");
				exit(1);
			}
			seqfilename = argv[par];
			par++;
		} else if (strcmp(argv[par],"-to") == 0) {
			par++;
			if (par >= argc) {
				printf("Missing to value!\n");
				exit(1);
			}
			input->to = atof(argv[par]);
			par++;
		} else if (strcmp(argv[par],"-alpha") == 0) {
			par++;
			if (par >= argc) {
				printf("Missing alpha value!\n");
				exit(1);
			}
			input->alpha = atof(argv[par]);
			par++;
		} else if (strcmp(argv[par],"-k") == 0) {
			par++;
			if (par >= argc) {
				printf("Missing k value!\n");
				exit(1);
			}
			input->k = atof(argv[par]);
			par++;
		} else if (strcmp(argv[par],"-sd") == 0) {
			par++;
			if (par >= argc) {
				printf("Missing seed value!\n");
				exit(1);
			}
			input->sd = atoi(argv[par]);
			par++;
		}else{
			printf("WARNING: unrecognized parameter '%s'!\n",argv[par]);
			par++;
		}
	}

	

	//
	// Legge i dati e verifica la correttezza dei parametri
	//
	if(seqfilename == NULL || strlen(seqfilename) == 0){
		printf("Missing ds file name!\n");
		exit(1);
	}


	input->seq = load_seq(seqfilename, &input->N, &d);
	

	
	if(d != 1){
		printf("Invalid size of sequence file, should be %ix1!\n", input->N);
		exit(1);
	} 

	if(input->to <= 0){
		printf("Invalid value of to parameter!\n");
		exit(1);
	}

	if(input->k <= 0){
		printf("Invalid value of k parameter!\n");
		exit(1);
	}

	if(input->alpha <= 0){
		printf("Invalid value of alpha parameter!\n");
		exit(1);
	}

	input->phi = alloc_matrix(input->N, 1);
	input->psi = alloc_matrix(input->N, 1);
	// Impostazione seed 
	srand(input->sd);
	// Inizializzazione dei valori
	gen_rnd_mat(input->phi, input->N);
	gen_rnd_mat(input->psi, input->N);

	//
	// Visualizza il valore dei parametri
	//

	if(!input->silent){
		printf("Dataset file name: '%s'\n", seqfilename);
		printf("Sequence lenght: %d\n", input->N);
	}

	// COMMENTARE QUESTA RIGA!
	//prova(input);
	//

	//
	// Predizione struttura terziaria
	//
	t = clock();
	pst(input);
	t = clock() - t;
	time = ((float)t)/CLOCKS_PER_SEC;

	if(!input->silent)
		printf("PST time = %.3f secs\n", time);
	else
		printf("%.3f\n", time);

	//
	// Salva il risultato
	//
	sprintf(fname_phi, "out32_%d_%d_phi.ds2", input->N, input->sd);
	save_out(fname_phi, input->phi, input->N);
	sprintf(fname_psi, "out32_%d_%d_psi.ds2", input->N, input->sd);
	save_out(fname_psi, input->psi, input->N);
	if(input->display){
		if(input->phi == NULL || input->psi == NULL)
			printf("out: NULL\n");
		else{
			int i,j;
			printf("energy: %f, phi: [", input->e);
			for(i=0; i<input->N; i++){
				printf("%f,", input->phi[i]);
			}
			printf("]\n");
			printf("psi: [");
			for(i=0; i<input->N; i++){
				printf("%f,", input->psi[i]);
			}
			printf("]\n");
		}
	}

	

	if(!input->silent)
		printf("\nDone.\n");

	dealloc_matrix(input->phi);
	dealloc_matrix(input->psi);
	free(input);

	return 0;
}
