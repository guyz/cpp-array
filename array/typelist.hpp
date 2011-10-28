/*
 * Copyright (C) 2011 by Alejandro M. Aragón
 * Written by Alejandro M. Aragón <alejandro.aragon@gmail.com>
 * All Rights Reserved
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

//! \file typelist.hpp
//
//  Taken from Alexandrescu, Andrei. "Modern C++ Design: Generic Programming and
//  Design Patterns Applied". Copyright (c) 2001. Addison-Wesley.
//

#ifndef ARRAY_TYPELIST_H
#define ARRAY_TYPELIST_H

namespace typelist {
  

struct NullType {};

template <class T, class U>
struct TypeList {
    typedef T Head;
    typedef U Tail;
};

template <
typename T1  = NullType, typename T2  = NullType, typename T3  = NullType,
typename T4  = NullType, typename T5  = NullType, typename T6  = NullType,
typename T7  = NullType, typename T8  = NullType, typename T9  = NullType,
typename T10 = NullType, typename T11 = NullType, typename T12 = NullType,
typename T13 = NullType, typename T14 = NullType, typename T15 = NullType,
typename T16 = NullType, typename T17 = NullType, typename T18 = NullType,
typename T19 = NullType, typename T20 = NullType
>
struct MakeTypeList
{
private:
    typedef typename MakeTypeList
    <
    T2 , T3 , T4 , 
    T5 , T6 , T7 , 
    T8 , T9 , T10, 
    T11, T12, T13,
    T14, T15, T16, 
    T17, T18, T19, T20
    >
    ::result TailResult;
    
public:
    typedef TypeList<T1, TailResult> result;
};

template<>
struct MakeTypeList<> {
    typedef NullType result;
};

template <class TList, unsigned int index> struct TypeAt;

template <class Head, class Tail>
struct TypeAt<TypeList<Head, Tail>, 0>
{
    typedef Head Result;
};

template <class Head, class Tail, unsigned int i>
struct TypeAt<TypeList<Head, Tail>, i>
{
    typedef typename TypeAt<Tail, i - 1>::Result Result;
};

template <class TList, class T> struct IndexOf;

template <class T>
struct IndexOf<NullType, T>
{
    enum { value = -1 };
};

template <class T, class Tail>
struct IndexOf<TypeList<T, Tail>, T>
{
    enum { value = 0 };
};

template <class Head, class Tail, class T>
struct IndexOf<TypeList<Head, Tail>, T>
{
private:
    enum { temp = IndexOf<Tail, T>::value };
public:
    enum { value = (temp == -1 ? -1 : 1 + temp) };
};
    
    
} /* namespace typelist */


#endif /* ARRAY_TYPELIST_H */
