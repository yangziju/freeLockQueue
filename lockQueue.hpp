#ifndef _LOCK_QUEUE_
#define _LOCK_QUEUE_

#include <mutex>

template<typename T>
class lockQueue {
public:
    lockQueue(unsigned long size);
    ~lockQueue();

    bool push(const T &data);
    bool pop(T &data);

private:
    unsigned long m_front;
    unsigned long m_back;

    unsigned long m_capacity;
    std::mutex m_lock;
    pthread_mutex_t m_mutex;
    T* m_ringBuff;
    inline unsigned long toIdx(unsigned long idx);
};

template<typename T>
lockQueue<T>::lockQueue(unsigned long size):
    m_back(0), m_front(0),
    m_capacity(size + 1),
    m_mutex(PTHREAD_MUTEX_INITIALIZER) {
    m_ringBuff = new T[size + 1];
}

template<typename T>
lockQueue<T>::~lockQueue() {
    delete[] m_ringBuff;
}

template<typename T>
bool lockQueue<T>::push(const T &data) {
    std::lock_guard<std::mutex> gd(m_lock);
    // pthread_mutex_lock(&m_mutex);
    if (toIdx(m_back + 1) == m_front) {
        // pthread_mutex_unlock(&m_mutex);
        return false;
    }
    m_ringBuff[m_back] = data;
    m_back = toIdx(m_back + 1);
    // pthread_mutex_unlock(&m_mutex);
    return true;
}

template<typename T>
bool lockQueue<T>::pop(T &data) {
    std::lock_guard<std::mutex> gd(m_lock);
    // pthread_mutex_lock(&m_mutex);
    if (m_front == m_back) {
        // pthread_mutex_unlock(&m_mutex);
        return false;
    }
    data = m_ringBuff[m_front];
    m_front = toIdx(m_front + 1);
    // pthread_mutex_unlock(&m_mutex);
    return true;
}

template<typename T>
unsigned long lockQueue<T>::toIdx(unsigned long idx) {
    return idx % m_capacity;
}

#endif // _LOCK_QUEUE_