//
// Created by Wassim Omais on 5/5/21.
//

#ifndef FASTFFT_COMPLEX_H
#define FASTFFT_COMPLEX_H

template <typename U>
struct complex {
    U dat[2];
    U real() { return dat[0]; }
    U imag() { return dat[1]; }
    complex() {
        dat[0] = dat[1] = 0.0;
    }
    template <typename T>
    complex(const T& o) { dat[0] = o, dat[1] = 0.0; }
    complex(U x, U y) { dat[0] = x, dat[1] = y; }
    complex operator+(const complex& o) {
        return complex(dat[0] + o.dat[0], dat[1] + o.dat[1]);
    }
    complex operator-(const complex& o) {
        return complex(dat[0] - o.dat[0], dat[1] - o.dat[1]);
    }
    complex operator*(const complex& o) {
        return complex(dat[0] * o.dat[0] - dat[1] * o.dat[1], dat[0] * o.dat[1] + dat[1] * o.dat[0]);
    }
    complex& operator*=(const complex& o) {
        return *this = *this * o;
    }
    template <typename T>
    complex& operator/=(const T& val) {
        dat[0] /= val;
        dat[1] /= val;
        return *this;
    }
};

#endif //FASTFFT_COMPLEX_H
