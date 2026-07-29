#include <stdio.h>
#include <time.h>

int main() {
    time_t start = time(NULL);
    while (time(NULL) - start < 3 * 60) {  // 8 minutes = 480 seconds
        asm volatile("nop");
    }
    return 0;
}

