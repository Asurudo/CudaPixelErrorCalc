//4次元ベクトルクラス
#pragma once

#ifndef VEC4_HPP
#define VEC4_HPP

#include<cmath>
#include<cassert>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//ベクトルクラス
class vec4 {
public:
    //コンストラクタ　
	vec4() : x( 0.0 ), y( 0.0 ), z( 0.0 ), w( 0.0 ) {}
    vec4( const float _x, const float _y, const float _z, const float _w ) : x( _x ), y( _y ), z( _z ), w( _w ) {}
    explicit vec4( const float s ) : x( s ), y( s ), z( s ), w( s ) {}
    //算術演算子
    inline vec4 operator+( const vec4 &rhs ) const { return vec4( x + rhs.x, y + rhs.y, z + rhs.z, w + rhs.w ); }
    inline vec4 operator-( const vec4 &rhs ) const { return vec4( x - rhs.x, y - rhs.y, z - rhs.z, w - rhs.w ); }
    inline vec4 operator*( const vec4 &rhs ) const { return vec4( x * rhs.x, y * rhs.y, z * rhs.z, w * rhs.w ); }
    inline vec4 operator/( const vec4 &rhs ) const { return vec4( x / rhs.x, y / rhs.y, z / rhs.z, w / rhs.w ); }
    inline vec4 operator*( const float scale ) const { return vec4( scale * x, scale * y, scale * z, scale * w ); }
    inline vec4 operator/( const float scale ) const { return vec4( x / scale, y / scale, z / scale, w / scale ); }

    inline vec4& operator+=( const vec4 &rhs ) { return x += rhs.x, y += rhs.y, z += rhs.z, w += rhs.w, *this; }
    inline vec4& operator-=( const vec4 &rhs ) { return x -= rhs.x, y -= rhs.y, z -= rhs.z, w -= rhs.w, *this; }
    inline vec4& operator*=( const vec4 &rhs ) { return x *= rhs.x, y *= rhs.y, z *= rhs.z, w *= rhs.w, *this; }
    inline vec4& operator*=( const float scale ) { return x *= scale, y *= scale, z *= scale, w *= scale, *this; }
    inline vec4& operator/=( const vec4 &rhs ) { return x /= rhs.x, y /= rhs.y, z /= rhs.z, w /= rhs.w, *this; }
    inline vec4& operator/=( const float scale ) { return x /= scale, y /= scale, z /= scale, w /= scale, *this; }
    inline vec4 operator-() const { return vec4( - x, - y, - z, - w ); }
    inline friend vec4 operator*( const float scale, const vec4 &v ) { return vec4( scale * v.x, scale * v.y, scale * v.z, scale * v.w ); }
    //アクセッサ
	inline float operator[]( const int dim ) const 
    { 
        return assert( 0 <= dim && dim < 4 ), ( dim == 0 ) ? x : ( dim == 1 ) ? y : ( dim == 2 ) ? z : w; 
    }
    //アクセッサ
	inline float& operator[]( const int dim ) 
    { 
        return assert( 0 <= dim && dim < 4 ), ( dim == 0 ) ? x : ( dim == 1 ) ? y : ( dim == 2 ) ? z : w; 
    }
    //内積/L2ノルム/正規化/外積
    inline friend float dot( const vec4 &lhs, const vec4 &rhs ) { return ( lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z + lhs.w * rhs.w ); }
    inline friend float norm( const vec4 &v ) { return sqrt( v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w ); }
    inline friend vec4 normalize( const vec4 &v ) 
    { 
        const float l = norm( v ); assert( std::abs( l ) > 1e-10 ); return vec4( v.x / l, v.y / l, v.z / l, v.w / l ); 
    }
    inline friend vec4 cross( const vec4 &lhs, const vec4 &rhs ) 
    { 
        return vec4( lhs.y * rhs.z - lhs.z * rhs.y, lhs.z * rhs.x - lhs.x * rhs.z, lhs.x * rhs.y - lhs.y * rhs.x, 0.f );
    }
    //ベクトルの要素が有限か否か
    inline friend bool isfinite( const vec4 &v ) 
    { 
        return std::isfinite( v.x ) && std::isfinite( v.y ) && std::isfinite( v.z ) && std::isfinite( v.w ); 
    }
    //最小・最大
    inline friend vec4 min( const vec4 &lhs, const vec4 &rhs )
    { 
        return vec4( std::min( lhs.x, rhs.x ), std::min( lhs.y, rhs.y ), std::min( lhs.z, rhs.z ), std::min( lhs.w, rhs.w ) ); 
    }
    inline friend vec4 max( const vec4 &lhs, const vec4 &rhs )
    { 
        return vec4( std::max( lhs.x, rhs.x ), std::max( lhs.y, rhs.y ), std::max( lhs.z, rhs.z ), std::max( lhs.w, rhs.w ) ); 
    }

    float x, y, z, w;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
typedef vec4 col4;

#endif //vec4_HPP
