
#pragma once

#ifndef NS_RENDERER_UTILITY_PSEUDO_COLOR_HPP
#define NS_RENDERER_UTILITY_PSEUDO_COLOR_HPP

#include<vector>
#include<array>

#include"vec4.hpp"


///////////////////////////////////////////////////////////////////////////////////////////////////
//pseudocolor
///////////////////////////////////////////////////////////////////////////////////////////////////

class pseudocolor
{
public:

	//コンストラクタ
	pseudocolor();
	pseudocolor(const float min, const float max);
	pseudocolor(const float min, const float max, std::vector<col4> &&cols);

	//擬似カラーを返す
	col4 operator()(const float x) const;

	//MATLABカラーバー
	static pseudocolor autumn(const float min, const float max);
	static pseudocolor bone(const float min, const float max);
	static pseudocolor cool(const float min, const float max);
	static pseudocolor copper(const float min, const float max);
	//static pseudocolor turbo(const float min, const float max);
	static pseudocolor hot(const float min, const float max);
	static pseudocolor hsv(const float min, const float max);
	static pseudocolor jet(const float min, const float max);
	static pseudocolor parula(const float min, const float max);
	static pseudocolor pink(const float min, const float max);
	static pseudocolor spring(const float min, const float max);
	static pseudocolor summer(const float min, const float max);
	static pseudocolor winter(const float min, const float max);

private:
	
	float m_min;
	float m_max;
	float m_interval;
	std::vector<col4> m_cols;
};


#include"pseudocolor-impl.hpp"

///////////////////////////////////////////////////////////////////////////////////////////////////

#endif