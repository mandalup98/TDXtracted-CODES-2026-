#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <asm/msr.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <sched.h>
#include <sys/types.h>
#include <time.h>
#include <errno.h>

#define IA32_PERFEVTSEL0  0x186  // Event select
#define IA32_PMC0         0xC1  // perfmon select

void initialize(int cpu, int *fd) {
    char msr_file_name[64];
    snprintf(msr_file_name, sizeof(msr_file_name), "/dev/cpu/%d/msr", cpu);
    *fd = open(msr_file_name, O_RDWR);
    if (*fd < 0) {
        perror("open");
        exit(1);
    }
}

void teardown(int fd) {
    close(fd);
}

void write_msr(int fd, uint32_t msr, uint64_t value) {
    if (pwrite(fd, &value, sizeof(value), msr) == -1) {
        perror("pwrite");
    }
}

uint64_t read_msr(int fd, uint32_t msr) {
    uint64_t value;
    if (pread(fd, &value, sizeof(value), msr) == -1) {
        perror("pread");
    }
    return value;
}

void configure_uncore_event(int fd, int counter) {

    uint64_t event_select_msr = IA32_PERFEVTSEL0 + counter;
    uint64_t event_value = 0;
    event_value |= (0x24 << 0);    // Event select
    event_value |= (0xFF << 8);     // UMask   
    event_value |= (1ULL << 16);    // user mode counting
    event_value |= (0ULL << 17);    // OS mode counting
    event_value |= (0ULL << 18);    // Edge detection enabled (0 = disabled)
    event_value |= (0ULL << 19);    // Pin control (0 = disabled)
    event_value |= (0ULL << 20);    // Interrupt (0 = disabled)
    event_value |= (1ULL << 22);    // Enable (keep this 1 always)
    event_value |= (0ULL << 23);    // Invert if required
    event_value |= (0x00 << 24);    // CMask

    write_msr(fd, event_select_msr, event_value);
}

void reset_uncore_counter(int fd, int counter) {
    uint64_t pmc_msr = IA32_PMC0 + counter;
    write_msr(fd, pmc_msr, 0);
}

uint64_t read_uncore_counter(int fd, int counter) {
    uint64_t pmc_msr = IA32_PMC0 + counter;
    return read_msr(fd, pmc_msr);
}

int main() {
    int intr;

    int cpu = 0;
    int fd;
    initialize(cpu, &fd);
    int counter = 0;
    uint64_t counter_value = 0;
    struct timespec req;
    req.tv_sec = 1;
    req.tv_nsec = 0;

    req.tv_nsec = 0;
    configure_uncore_event(fd, counter);
    reset_uncore_counter(fd, counter);
    int traces = 0;
    int counter_ = 0;
    srand((unsigned int) time(NULL));
    while(traces <= 150){
        traces++;
        counter_value = read_uncore_counter(fd, counter);
        nanosleep(&req, NULL);
        //__asm__ __volatile__("FCOM ST(0)");
        /*
        for(int i = 0; i < 100000; i++){
                int x = rand();
                if (x % 2 == 0) {
                        ++counter_;
                } else{
                        --counter_;
                }
        }*/

        printf("%lu\n", (unsigned long)(read_uncore_counter(fd, counter) - counter_value));
        fflush(stdout);
    }
    teardown(fd);
    return 0;
}


