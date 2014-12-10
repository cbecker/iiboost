#ifndef _SMART_PTRS_H_
#define _SMART_PTRS_H_

/** Some smart pointer shortcuts, C++11 **/
#include <memory>


// shared ptr that won't delete pointer
// useful for passing references as pointers
// For example:
//		MyClass obj;
//		shared_ptr_nodelete( MyClass, &obj )
//
#define shared_ptr_nodelete(T, ptr) \
	std::shared_ptr<T>( ptr, [](void *){} )


#endif
