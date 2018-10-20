//
// Created by lut on 18-10-20.
//

#include "accum_conv.h"

namespace suo15features{

    Accum_Conv::Accum_Conv():w(0.0f) {}

    Accum_Conv::Accum_Conv(float init): v(init), w(0.0f){}

    void Accum_Conv::add(float value, float weight) {
        this->v += value*weight;
        this->w += weight;
    }

    void Accum_Conv::sub(float value, float weight) {
        this->v -= value*weight;
        this->w -= weight;
    }

    float Accum_Conv::normalized(){
        return this->v/this->w;
    }

    float Accum_Conv::normalized(float weight) {
        return this->v/weight;
    }

}