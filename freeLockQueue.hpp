#ifndef _FREE_LOCK_QUEUE_
#define _FREE_LOCK_QUEUE_

#include <sched.h> // sched_yield()

#define QUEUE_DEFAULT_SIZE 30000001
#define QINT unsigned long

#define CAS(a_ptr, a_oldVal, a_newVal) __sync_bool_compare_and_swap(a_ptr, a_oldVal, a_newVal)
#define atomicAdd(a_ptr, a_count) __sync_fetch_and_add (a_ptr, a_count)
#define atomicSub(a_ptr, a_count) __sync_fetch_and_sub (a_ptr, a_count)

template<typename T, QINT Q_SIZE = QUEUE_DEFAULT_SIZE>
class freeLockQueue {
public:
    freeLockQueue();
    ~freeLockQueue();

    QINT get_count();
    bool push(const T &data);
    bool pop(T &data);
    bool try_dequeue(T &data);

private:
    volatile QINT m_count;          // 队列中元素个数
    volatile QINT m_readIdx;        // 可出队元素的索引位置
    volatile QINT m_writeIdx;       // 可入队元素的索引位置
    volatile QINT m_maxReadIdx;     // 最大可读元素索引的下一个位置

    T* m_ringBuff;                  // 保存数据的ringBuff
    inline QINT toIdx(QINT index);

};

template<typename T, QINT Q_SIZE>
freeLockQueue<T, Q_SIZE>::freeLockQueue():
                m_count(0), m_readIdx(0), 
                m_writeIdx(0), m_maxReadIdx(0) {
    m_ringBuff = new T[Q_SIZE];
}

template<typename T, QINT Q_SIZE>
freeLockQueue<T, Q_SIZE>::~freeLockQueue() {
    delete[] m_ringBuff;
}

template<typename T, QINT Q_SIZE>
bool freeLockQueue<T, Q_SIZE>::push(const T &data) {
    QINT currWriteIdx = m_writeIdx;

    // 先更新 ++m_writeIdx
    do {
        currWriteIdx = m_writeIdx;

        // ringBuff满了就不能再入队
        if(m_readIdx == toIdx(currWriteIdx + 1)) {
            return false;
        }

    }while(!CAS(&m_writeIdx, currWriteIdx, toIdx(currWriteIdx + 1)));

    m_ringBuff[currWriteIdx] = data;

    // 再更新m_maxReadIdx:
    // 当有多个线程都成功执行了上面第一个CAS操作还未执行第二个CAS操作时
    // 第二个CAS操作能够保证这些线程按照执行第一个CAS操作的先后顺序执行
    // 即：三个线程在执行第一个CAS操作的顺序为：T1 -> T2 -> T3
    // 则：他们执行第二个CAS操作的顺序一定也是：T1 -> T2 -> T3
    // 因为三个线程执行第一个CAS操作后currMaxReadIdx一定为：T1:OLD, T2:OLD+1, T3:(OLD+1)+1
    // 而m_maxReadIdx此时还没有更新，值都为：OLD, 所以CAS保证了每个生产者线程执行的先后顺序
    while(!CAS(&m_maxReadIdx, currWriteIdx, toIdx(currWriteIdx + 1))) {
        // 这里的yield是为了避免消费者线程数量大于硬件支持的总线程数时出现死循环
        // 比如当前系统只有一个core，同样上面上个现场，但是这里时T2先执行第二个CAS操作
        // 那么这里就会一直循环，而其它两个线程有抢不到CPU的情况
        sched_yield(); 
    }

    atomicAdd(&m_count, 1);

    return true;
}

template<typename T, QINT Q_SIZE>
bool freeLockQueue<T, Q_SIZE>::pop(T &data) {
    QINT currRreadIdx = m_readIdx;

    // 先更新 ++m_readIdx
    do {
        currRreadIdx = m_readIdx;

        // 队列为空了就不能出队
        if (currRreadIdx == m_maxReadIdx) {
            return false;
        }

        data = m_ringBuff[currRreadIdx];

    } while(!CAS(&m_readIdx, currRreadIdx, toIdx(currRreadIdx + 1)));

    atomicSub(&m_count, 1);

    return true;
}

template<typename T, QINT Q_SIZE>
bool freeLockQueue<T, Q_SIZE>::try_dequeue(T &data) {
    if (m_count == Q_SIZE - 1)
        return false;

    pop(data);

    return true;
}

template<typename T, QINT Q_SIZE>
QINT freeLockQueue<T, Q_SIZE>::get_count() {
    return m_count;
}

template<typename T, QINT Q_SIZE>
QINT freeLockQueue<T, Q_SIZE>::toIdx(QINT index) {
    return index % Q_SIZE;
}

#endif // _FREE_LOCK_QUEUE_