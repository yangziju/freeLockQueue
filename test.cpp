#include "freeLockQueue.hpp"
#include "lockQueue.hpp"
#include "arrayFreeLockQue.hpp"
#include <iostream>
#include <pthread.h>
#include <sys/time.h>
#include <assert.h>
using namespace std;

#define TIME_SUB_MS(s, e) ((e.tv_sec - s.tv_sec) * 1000 + (1.0 * (e.tv_usec - s.tv_usec)) / 1000)

unsigned long g_push_cnt = 0;
unsigned long g_pop_cnt = 0;
unsigned long g_fail_cnt = 0;
int g_producer_cnt = 0;
int g_consumer_cnt = 0;
int g_last_data;
unsigned long g_tasks_per_producer = 0;


typedef void* (*op_fn_t)(void *arg);
template<typename T>
void* producer_th(void *arg) {
    T* que = static_cast<T*>(arg);
    unsigned long success = 0, fail = 0;

    for (int i = 1; i <= g_tasks_per_producer; i++) {
        if (que->push(i)) {
            success++;
            atomicAdd(&g_push_cnt, 1);
        } else {
            fail++;
            atomicAdd(&g_fail_cnt, 1);
        }
    }
    printf("[producer:%ld] push failCount: %lu, successCount: %lu, success rate: %.0lf\%\n", 
        pthread_self(), fail, success, success * 1.0 / g_tasks_per_producer * 100);
    return NULL;
}

void do_work(const int loop)
{
    for (int i = 0; i < loop; i++) {

    }
}
template<typename T>
void* consumer_th(void *arg) {
    T* que = static_cast<T*>(arg);
    unsigned long success = 0, fail = 0;
    int data;
    while (true) {
        if (que->pop(data)) {
            atomicAdd(&g_pop_cnt, 1);
            success++;
            if (g_consumer_cnt == 1 && g_producer_cnt == 1) {
                if (g_last_data + 1 != data) {
                    printf("[concumer:%ld] error: last_val = %d, "
                        "curr_val = %d, curr_expected = %d\n",
                        pthread_self(), g_last_data,
                        data, g_last_data + 1);
                }
                g_last_data = data;
            }
            do_work(1000);
        } else {
            fail++;
        }
        if (g_pop_cnt == g_tasks_per_producer * g_producer_cnt - g_fail_cnt) {
            break;
        }
    }
    printf("[consumer:%ld] pop failCount: %lu, successCount: %lu, success rete: %.0lf\%\n", 
        pthread_self(), fail, success, 
        success * 1.0 / (g_tasks_per_producer * g_producer_cnt) * 100);

    return NULL;
}

template<typename T>
void queue_test_fn(op_fn_t producer, op_fn_t consumer, T &que, string str) {
    struct timeval begin_tv, end_tv;
    pthread_t *p_tid = NULL, *c_tid = NULL;
    int ret;
    g_last_data = 0;
    g_push_cnt = 0;
    g_pop_cnt = 0;
    g_fail_cnt = 0;
    p_tid = (pthread_t*)malloc(g_producer_cnt * sizeof(pthread_t));
    c_tid = (pthread_t*)malloc(g_consumer_cnt * sizeof(pthread_t));
    assert(p_tid && c_tid);

    gettimeofday(&begin_tv, NULL);

    for (int i = 0; i < g_producer_cnt; i++) {
        ret = pthread_create(&p_tid[i], NULL, producer, &que);
        assert(ret == 0);
    }
    for (int i = 0; i < g_consumer_cnt; i++) {
        ret = pthread_create(&c_tid[i], NULL, consumer, &que);
        assert(ret == 0);
    }

    for (int i = 0; i < g_producer_cnt; i++) {
        pthread_join(p_tid[i], NULL);
    }
    for (int i = 0; i < g_consumer_cnt; i++) {
        pthread_join(c_tid[i], NULL);
    }

    gettimeofday(&end_tv, NULL);
    long time = TIME_SUB_MS(begin_tv, end_tv);
    printf("[%s] producer:%d, consumer:%d, "
        "push:%d, pop:%d, time:%ld ms, ops:%.0lf\n\n", 
        str.c_str(), g_producer_cnt, g_consumer_cnt, 
        g_push_cnt, g_pop_cnt, 
        time, 1.0 * g_pop_cnt / time * 1000);

    if (p_tid) free(p_tid);
    if (c_tid) free(c_tid);
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        printf("%s [procuderNums] [consumerNums] [taskNums]\n", argv[0]);
        return 0;
    }
    g_producer_cnt = atoi(argv[1]);
    g_consumer_cnt = atoi(argv[2]);
    g_tasks_per_producer = atoi(argv[3]) / g_producer_cnt;

    lockQueue<int> lockQue(QUEUE_DEFAULT_SIZE);
    queue_test_fn<lockQueue<int>>(producer_th<lockQueue<int>>, 
                consumer_th<lockQueue<int>>, lockQue, "lockQueue");

    freeLockQueue<int> que;
    queue_test_fn<freeLockQueue<int>>(producer_th<freeLockQueue<int>>, 
                consumer_th<freeLockQueue<int>>, que, "freeLockQueue");
    
    ArrayLockFreeQueue<int> afq;
    queue_test_fn<ArrayLockFreeQueue<int>>(producer_th<ArrayLockFreeQueue<int>>, 
                consumer_th<ArrayLockFreeQueue<int>>, afq, "arrayFreeLockQueue");
    return 0;
}