#include <assert.h>
#include <stdio.h>

int* add(int *a, int *b, int *result, int N){
	
	int i;
	for( i = 0; i < N; i++){
		result[i] = a[i] + b[i];
	}

	return result;
}

void onHost(){
	
	const int ARRAY_SIZE = 10;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);
	int *a, *b, *result;

	a = (int*)malloc(ARRAY_BYTES);
	b = (int*)malloc(ARRAY_BYTES);
	result = (int*)malloc(ARRAY_BYTES);

	for (int i=0; i<ARRAY_SIZE; i++) {
        a[i] = -i;
        b[i] = i * i;
		result[i]=0;
    }

    add(a, b, result, ARRAY_SIZE);

    for (int i=0; i<ARRAY_SIZE; i++) {
		assert( a[i] + b[i] == result[i] );
    }

    printf("-: successful execution :-\n");

    free(a);
    free(b);
    free(result);

}


int main(){
	
	onHost();
	return 0;
}