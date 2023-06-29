#ifndef _LOCK_QUEUE_
#define _LOCK_QUEUE_

#include <mutex>

template<typename T>
class lockQueue {
public:
    lockQueue(unsigned long size);
    ~lockQueue();

    bool enqueue(const T &data);
    bool dequeue(T &data);

private:
    unsigned long m_front;
    unsigned long m_back;

    unsigned long m_capacity;
    std::mutex m_lock;
    T* m_ringBuff;
    inline unsigned long toIdx(unsigned long idx);
};

template<typename T>
lockQueue<T>::lockQueue(unsigned long size):
    m_back(0), m_front(0),
    m_capacity(size + 1) {
    m_ringBuff = new T[size + 1];
}

template<typename T>
lockQueue<T>::~lockQueue() {
    delete[] m_ringBuff;
}

template<typename T>
bool lockQueue<T>::enqueue(const T &data) {
    std::lock_guard<std::mutex> gd(m_lock);
    if (toIdx(m_back + 1) == m_front) {
        return false;
    }
    m_ringBuff[m_back] = data;
    m_back = toIdx(m_back + 1);
    return true;
}

template<typename T>
bool lockQueue<T>::dequeue(T &data) {
    std::lock_guard<std::mutex> gd(m_lock);
    if (m_front == m_back) {
        return false;
    }
    data = m_ringBuff[m_front];
    m_front = toIdx(m_front + 1);
    return true;
}

template<typename T>
unsigned long lockQueue<T>::toIdx(unsigned long idx) {
    return idx % m_capacity;
}

#endif // _LOCK_QUEUE_
