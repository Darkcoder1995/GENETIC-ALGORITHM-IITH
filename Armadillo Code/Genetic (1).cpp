#include <iostream>
#include <armadillo>
#include <sstream>
#include <time.h>
#include <fstream>
#include <vector>

using namespace std;
using namespace arma;

int score(mat Y , mat B , mat Z , mat B1 , int R);

int main(int argc, char const *argv[]) {

      time_t current_time_date;
      time(&current_time_date);
      string time_date = ctime(&current_time_date);

      wall_clock timer;

      // arma_rng::set_seed_random();
      arma_rng::set_seed(time (NULL));
      srand(time(NULL));

      imat dimension , no_of_labels , labels , no_of_kmeans;
      dimension.load("dimension.csv" , csv_ascii);
      no_of_labels.load("no_of_labels.csv" , csv_ascii);
      labels.load("labels.csv" , csv_ascii);
      no_of_kmeans.load("no_of_kmeans.csv" , csv_ascii);

      mat X , Y , X1 , Z;
      X.load("X.csv" , csv_ascii);
      Y.load("Y.csv" , csv_ascii);
      X1.load("K_means.csv" , csv_ascii);
      Z.load("Z.csv" , csv_ascii);

      X = X.t();
      X1 = X1.t();

      int X_M = X.n_rows;
      int X_N = X.n_cols;
      int X1_M = X1.n_rows;
      int X1_N = X1.n_cols;
      int R = 128;
      int D = X_M;

      int pool_size = 100;

      // cout << X.n_rows << "\t" << X.n_cols << endl;
      // cout << Y.n_rows << "\t" << Y.n_cols << endl;
      // cout << X1.n_rows << "\t" << X1.n_cols << endl;
      // cout << Z.n_rows << "\t" << Z.n_cols << endl;

      mat stats;

      mat parent_W[pool_size];
      for (size_t i = 0; i < pool_size; i++) {
            parent_W[i] = randi<mat>(R,D,distr_param(-1,1));
            parent_W[i] = sign(parent_W[i]);
      }


      int total_generation = 1;
      for (size_t generation = 0; generation <= total_generation; generation++) {
            timer.tic();
            printf("\nGENERATION %lu\t", generation);

            vec parent_score(pool_size);

            #pragma omp parallel for
            for (size_t i = 0; i < pool_size; i++) {
                  mat B = sign(parent_W[i]*X+0.0000000001);
                  mat B1 = sign(parent_W[i]*X1+0.0000000001);
                  parent_score(i) = score(Y , B , Z , B1 , R);
            }

            double mean_score = mean(parent_score)*1.0/X_N;
            double min_score = min(parent_score)*1.0/X_N;
            double max_score = max(parent_score)*1.0/X_N;
            double std_score = stddev(parent_score)*1.0/X_N;

            rowvec stats_row(4);
            stats_row(0) = min_score;
            stats_row(1) = mean_score;
            stats_row(2) = std_score;
            stats_row(3) = max_score;

            stats = join_cols(stats , stats_row);

            cout << min_score << "   " << mean_score << " +- " << std_score << "   " << max_score << "   time   " << timer.toc();


            uvec above_mean_index = find(parent_score >= mean_score);
            vec above_mean_score = parent_score(above_mean_index);
            mat above_mean_W[above_mean_score.n_rows];
            for (size_t i = 0; i < above_mean_score.n_rows; i++) {
                  above_mean_W[i] = parent_W[above_mean_index(i)];
            }

            ivec percent = conv_to<ivec>::from((above_mean_score/sum(above_mean_score))*above_mean_score.n_rows*10);

            imat new_pool;
            for (size_t i = 0; i < percent.n_rows; i++) {
                  int per = percent(i);
                  imat temp(per,1);
                  temp.fill(i);
                  new_pool = join_cols(new_pool , temp);
            }

            new_pool = shuffle(new_pool);

            int selected_index1;
            int selected_index2;


            // CROSSOVER 1 SELECTION
            selected_index1 = new_pool(rand()%new_pool.n_rows);
            selected_index2 = new_pool(rand()%new_pool.n_rows);
            while (selected_index1 == selected_index2) {
                  selected_index2 = new_pool(rand()%new_pool.n_rows);
            }

            // CROSSOVER 1    MULTIPOINT        around 25%
            mat C1_W1 = above_mean_W[selected_index1];
            mat C1_W2 = above_mean_W[selected_index2];

            #pragma omp parallel for
            for (size_t i = 0; i < 6; i++) {
                  uvec row_index = randi<uvec>(R*0.3 ,distr_param(0,R-1));
                  uvec col_index = randi<uvec>(D*0.3 ,distr_param(0,D-1));

                  C1_W1(row_index,col_index) = above_mean_W[selected_index2].submat(row_index,col_index);
                  C1_W2(row_index,col_index) = above_mean_W[selected_index1].submat(row_index,col_index);
            }


            // CROSSOVER 2 SELECTION
            selected_index1 = new_pool(rand()%new_pool.n_rows);
            selected_index2 = new_pool(rand()%new_pool.n_rows);
            while (selected_index1 == selected_index2) {
                  selected_index2 = new_pool(rand()%new_pool.n_rows);
            }

            // CROSSOVER 2    COL               around 25%
            mat C2_W1 = above_mean_W[selected_index1];
            mat C2_W2 = above_mean_W[selected_index2];

            for (size_t i = 0; i < 2; i++) {
                  uvec col_index = randi<uvec>(D*0.25 ,distr_param(0,D-1));

                  C2_W1.cols(col_index) = above_mean_W[selected_index2].cols(col_index);
                  C2_W2.cols(col_index) = above_mean_W[selected_index1].cols(col_index);
            }


            // CROSSOVER 3 SELECTION
            selected_index1 = new_pool(rand()%new_pool.n_rows);
            selected_index2 = new_pool(rand()%new_pool.n_rows);
            while (selected_index1 == selected_index2) {
                  selected_index2 = new_pool(rand()%new_pool.n_rows);
            }

            // CROSSOVER 3    SHUFFLE CROSS     around 25%
            mat C3_W1 = above_mean_W[selected_index1];
            mat C3_W2 = above_mean_W[selected_index2];

            #pragma omp parallel for
            for (size_t i = 0; i < 6; i++) {
                  uvec row_index = randi<uvec>(R*0.3 ,distr_param(0,R-1));
                  uvec col_index = randi<uvec>(D*0.3 ,distr_param(0,D-1));

                  mat temp_cross_W1 = shuffle(C3_W1);
                  mat temp_cross_W2 = shuffle(C3_W2);

                  C3_W2(row_index,col_index) = temp_cross_W1.submat(row_index,col_index);
                  C3_W1(row_index,col_index) = temp_cross_W2.submat(row_index,col_index);
            }



            // MUTATION 1 SELECTION
            selected_index1 = new_pool(rand()%new_pool.n_rows);

            // MUTATION 1    MULTIPOINT         around 10%
            mat M1_W1 = above_mean_W[selected_index1];

            #pragma omp parallel for
            for (size_t i = 0; i < 10; i++) {
                  uvec row_index = randi<uvec>(R*0.1 ,distr_param(0,R-1));
                  uvec col_index = randi<uvec>(D*0.1 ,distr_param(0,D-1));

                  M1_W1(row_index,col_index) = -above_mean_W[selected_index1].submat(row_index,col_index);
            }


            // MUTATION 2 SELECTION
            selected_index1 = new_pool(rand()%new_pool.n_rows);

            // MUTATION 2    +1                 around 10%
            mat M2_W1 = above_mean_W[selected_index1];

            #pragma omp parallel for
            for (size_t i = 0; i < 10; i++) {
                  uvec row_index = randi<uvec>(R*0.1 ,distr_param(0,R-1));
                  uvec col_index = randi<uvec>(D*0.1 ,distr_param(0,D-1));

                  M2_W1(row_index,col_index) = 1+above_mean_W[selected_index1].submat(row_index,col_index);
                  M2_W1.replace(2,-1);
            }


            // MUTATION 3 SELECTION
            selected_index1 = new_pool(rand()%new_pool.n_rows);

            // MUTATION 3    -1                 around 10%
            mat M3_W1 = above_mean_W[selected_index1];

            #pragma omp parallel for
            for (size_t i = 0; i < 10; i++) {
                  uvec row_index = randi<uvec>(R*0.1 ,distr_param(0,R-1));
                  uvec col_index = randi<uvec>(D*0.1 ,distr_param(0,D-1));

                  M3_W1(row_index,col_index) = -1+above_mean_W[selected_index1].submat(row_index,col_index);
                  M3_W1.replace(-2,1);
            }



            // ELITE SELECTION
            int elite_index = index_max(above_mean_score);
            mat elite_W = above_mean_W[elite_index];



            // RANDOM MAT GENERATION 3%

            mat R_W1 = randi<mat>(R,D,distr_param(-1,1));
            mat R_W2 = randi<mat>(R,D,distr_param(-1,1));
            mat R_W3 = randi<mat>(R,D,distr_param(-1,1));

            R_W1 = sign(R_W1);
            R_W2 = sign(R_W2);
            R_W3 = sign(R_W3);



            ////////////////////////////////////////////////////////////////////

            mat child_W[pool_size];

            int k=0;

            child_W[k] = elite_W;
            k++;

            for (size_t i = 0; i < 15; i++) {
                  child_W[k] = C1_W1;
                  k++;
            }
            for (size_t i = 0; i < 15; i++) {
                  child_W[k] = C1_W2;
                  k++;
            }
            for (size_t i = 0; i < 15; i++) {
                  child_W[k] = C2_W1;
                  k++;
            }for (size_t i = 0; i < 15; i++) {
                  child_W[k] = C2_W2;
                  k++;
            }
            for (size_t i = 0; i < 15; i++) {
                  child_W[k] = C3_W1;
                  k++;
            }
            for (size_t i = 0; i < 15; i++) {
                  child_W[k] = C3_W2;
                  k++;
            }

            for (size_t i = 0; i < 2; i++) {
                  child_W[k] = M1_W1;
                  k++;
            }
            for (size_t i = 0; i < 2; i++) {
                  child_W[k] = M2_W1;
                  k++;
            }
            for (size_t i = 0; i < 2; i++) {
                  child_W[k] = M3_W1;
                  k++;
            }

            child_W[k] = R_W1;
            k++;
            child_W[k] = R_W2;
            k++;
            child_W[k] = R_W3;
            k++;

            uvec random_shuffle_index(pool_size);
            for (size_t i = 0; i < pool_size; i++) {
                  random_shuffle_index(i) = i;
            }
            random_shuffle_index = shuffle(random_shuffle_index);

            for (size_t i = 0; i < pool_size; i++) {
                  parent_W[i] = child_W[random_shuffle_index(i)];
            }


            if(generation == total_generation){
                  string name = "W " + time_date + ".csv";
                  elite_W.save(name , csv_ascii);
            }
      }

      string name = "stats " + time_date + ".csv" ;
      stats.save(name , csv_ascii);


      return 0;
}

int score(mat Y , mat B , mat Z , mat B1 , int R){
      int score_W=0;
      mat B1_B = (B1.t())*B;
      mat hamming_distance = 0.5*(R-B1_B);
      for (size_t i = 0; i < hamming_distance.n_cols; i++) {
            uword min_index = hamming_distance.col(i).index_min();
            if (Y(i,0) == Z(min_index,0)) {
                  score_W++;
            }
      }
      // cout << "score_W" << '\t' << score_W*1.0/hamming_distance.n_cols << endl;
      return score_W;
}

// cout << C1_W1.n_rows << '\t' << C1_W1.n_cols << endl;
