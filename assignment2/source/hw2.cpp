#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <curses.h>
#include <termios.h>
#include <fcntl.h>

#define ROW 10
#define COLUMN 50 
#define LOGLENGTH 15
#define LOGSPEED 500000
#define MAXINTERVAL 5

pthread_mutex_t mutex;

struct Node{
	int x , y; 
	Node( int _x , int _y ) : x( _x ) , y( _y ) {}; 
	Node(){} ; 
} frog ; 


char map[ROW+10][COLUMN];

//flag = 0 for normal status, 1 for lose, 2 for win, 4 for quit
int flag;

void printMap(void);


// Determine a keyboard is hit or not. If yes, return 1. If not, return 0. 
int kbhit(void){
	struct termios oldt, newt;
	int ch;
	int oldf;

	tcgetattr(STDIN_FILENO, &oldt);

	newt = oldt;
	newt.c_lflag &= ~(ICANON | ECHO);

	tcsetattr(STDIN_FILENO, TCSANOW, &newt);
	oldf = fcntl(STDIN_FILENO, F_GETFL, 0);

	fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

	ch = getchar();

	tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
	fcntl(STDIN_FILENO, F_SETFL, oldf);

	if(ch != EOF)
	{
		ungetc(ch, stdin);
		return 1;
	}
	return 0;
}


void *logs_move( void *t ){

	int logId = (int) t;

	/*  Move the logs  */
	if(!logId){
        int randomInterval[ROW];
        bool crossBoun = false;
        char orignalRow[COLUMN];
        unsigned int seedNum = 0;
		while(!flag){
			usleep(LOGSPEED);
            srand(seedNum++);
            for(int a = 1; a < ROW; a++)
                randomInterval[a] = rand() % MAXINTERVAL + 1;

			//check whether frog on the logs reaching the left or right boundaries
            crossBoun = false;

			pthread_mutex_lock(&mutex);

			//change the frog position on the logs
			if(frog.x!=ROW && frog.x!=0){
				if(frog.x%2) frog.y -= randomInterval[frog.x];
				else frog.y += randomInterval[frog.x];

                if(frog.y < 0 || frog.y >= COLUMN-1)
                    crossBoun = true;
			}

            if(crossBoun){
                flag = 1;
                pthread_mutex_unlock(&mutex);
                break;
            }

			for(int j = 1; j < ROW; j++){

                for(int a = 0; a < COLUMN; a++){
                    orignalRow[a] = map[j][a];
                }

				//left shift
				if(j%2){
					for(int i = 0; i < COLUMN-1; i++){
						map[j][i] = orignalRow[(i+randomInterval[j]+COLUMN-1)%(COLUMN-1)];
					}
				}

				//right shift
				else{
					for(int i = 0; i < COLUMN-1; i++){
						map[j][i] = orignalRow[(i-randomInterval[j]+COLUMN-1)%(COLUMN-1)];
					}
				}
			}
			printMap();
			pthread_mutex_unlock(&mutex);
		}
		pthread_exit(NULL);
	}
	else{
		char lastStep = '|';
		bool isChange = false;
		while(!flag){

			pthread_mutex_lock(&mutex);

			/*  Check keyboard hits, to change frog's position or quit the game. */
			if(kbhit()){
				
				char currentStep;
				char dir = getchar();
				int curX = frog.x;
				int curY = frog.y;

				if(dir == 27){
					while (kbhit()!=0)
					{
						dir = getchar();
					}
					pthread_mutex_unlock(&mutex);
					continue;
				}

				isChange = false;

				if( dir == 'q' || dir == 'Q' ){	
					flag = 4;
					pthread_mutex_unlock(&mutex);
					break;
				}
				
				if( dir == 'w' || dir == 'W' ){
					frog.x--;
					isChange = true;
				}
	 
				if( dir == 'a' || dir == 'A' ){
					
					//leftmost square on the bank
					//does not change the position
					if(!(frog.x==ROW && frog.y==0)){
						frog.y--;
						isChange = true;
					}

					if(frog.x!=ROW && frog.y==0){
						flag = 1;
						pthread_mutex_unlock(&mutex);
						break;
					}
				}

				if( dir == 'd' || dir == 'D' ){
					
					//leftmost square on the bank
					//dose not change position
					if(!(frog.x==ROW && frog.y==COLUMN-2)){
						frog.y++;
						isChange = true;
					}

					if(frog.x!=ROW && frog.y==COLUMN-2){
						flag = 1;
						pthread_mutex_unlock(&mutex);
						break;
					}
				}			

				if( dir == 's' || dir == 'S' ){

					//on the bank
					//does not change the position
					if(frog.x!=ROW){
						frog.x++;
						isChange = true;
					}
				}

				//move the frog
				map[curX][curY] = lastStep;
				lastStep = map[frog.x][frog.y];
				map[frog.x][frog.y] = '0';

				/*  Check game's status  */

				//lose the game
				if(lastStep==' '){
					flag = 1;
					pthread_mutex_unlock(&mutex);
					break;
				}
				else if(frog.x==0){
					flag = 2;
					pthread_mutex_unlock(&mutex);
					break;
				}

				/*  Print the map on the screen  */
				if(isChange) printMap();
			}
			pthread_mutex_unlock(&mutex);	
		}
		pthread_exit(NULL);
	}
}

void logsInitialization(void){
	int startPoint;
	for(int i = 1; i<ROW; i++){
		startPoint = rand() % COLUMN;
		for(int j = 0; j<LOGLENGTH; j++){
			map[i][(startPoint+j)%(COLUMN-1)] = '=';
		}
	}
}

void printMap(void){
	printf("\033[H\033[2J");
	for(int i = 0; i <= ROW; ++i)	
		puts( map[i] );
}

int main( int argc, char *argv[] ){

	flag = 0;
	srand(time(NULL));
	pthread_t threadLogs, threadFrog;
	pthread_mutex_init(&mutex, NULL);

	// Initialize the river map and frog's starting position
	memset( map , 0, sizeof( map ) ) ;
	int i, j, rc1, rc2; 
	for( i = 1; i < ROW; ++i ){	
		for( j = 0; j < COLUMN - 1; ++j )	
			map[i][j] = ' ' ;  
	}	

	for( j = 0; j < COLUMN - 1; ++j )	
		map[ROW][j] = map[0][j] = '|' ;

	for( j = 0; j < COLUMN - 1; ++j )	
		map[0][j] = map[0][j] = '*' ;

	frog = Node( ROW, (COLUMN-1) / 2 ) ; 
	map[frog.x][frog.y] = '0' ; 

	logsInitialization();

	//Print the map into screen
	printMap();

	/*  Create pthreads for wood move and frog control.  */
	rc1 = pthread_create(&threadLogs, NULL, logs_move, (void*) 0);
	rc2 = pthread_create(&threadFrog, NULL, logs_move, (void*) 1);
	if(rc1){
		printf("ERROR: return code from pthread_create() is %d", rc1);
		exit(1);
	}
	if(rc2){
		printf("ERROR: return code from pthread_create() is %d", rc2);
		exit(1);
	}

	/* Wait for game to end */
	pthread_join(threadLogs, NULL);
	pthread_join(threadFrog, NULL);

	/*  Display the output for user: win, lose or quit.  */
	printf("\033[H\033[2J");
	if(flag==1){
		printf("You lose the game!!\n");
	}
	else if(flag==2){
		printf("You win the game!!\n");
	}
	else{
		printf("You exit the game.\n");
	}

	pthread_mutex_destroy(&mutex);
	pthread_exit(NULL);
	return 0;

}