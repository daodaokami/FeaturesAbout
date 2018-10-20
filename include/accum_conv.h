//
// Created by lut on 18-10-20.
//

#ifndef LUT15VO_ACCUM_H
#define LUT15VO_ACCUM_H

namespace suo15features{
    class Accum_Conv{
    public:
        float v;
        float w;

    public:
        Accum_Conv(void);

        Accum_Conv(float init);

        void add(float value, float weight);

        void sub(float value, float weight);

        float normalized(float weight);

        float normalized(void);
    };

}

#endif //LUT15VO_ACCUM_H
