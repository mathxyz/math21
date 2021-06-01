///* Copyright 2015 The math21 Authors. All Rights Reserved.
//
//Licensed under the Apache License, Version 2.0 (the "License");
//you may not use this file except in compliance with the License.
//You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//Unless required by applicable law or agreed to in writing, software
//distributed under the License is distributed on an "AS IS" BASIS,
//WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//See the License for the specific language governing permissions and
//limitations under the License.
//==============================================================================*/
//
//#pragma once
//
//#include <string>
//#include <new>          // for std::bad_alloc
//#include <iostream>
//#include <cassert>
//#include <cstdlib>
//#include <exception>
//
//namespace math21 {
//
//    enum error_type {
//        EUNSPECIFIED,
//        EFATAL,
//        EBROKEN_ASSERT
//    };
//
//    // the base exception class
//    class m21error : public std::exception {
//        /*!
//            WHAT THIS OBJECT REPRESENTS
//                This is the base exception class for the math21 library.  i.e. all
//                exceptions in this library inherit from this class.
//        !*/
//
//    public:
//        m21error(
//                error_type t,
//                const std::string &a
//        ) : info(a), type(t) {}
//
//        /*!
//            ensures
//                - #type == t
//                - #info == a
//        !*/
//
//        m21error(
//                error_type t
//        ) : type(t) {}
//
//        /*!
//            ensures
//                - #type == t
//                - #info == ""
//        !*/
//
//        m21error(
//                const std::string &a
//        ) : info(a), type(EUNSPECIFIED) {}
//
//        /*!
//            ensures
//                - #type == EUNSPECIFIED
//                - #info == a
//        !*/
//
//        m21error(
//        ) : type(EUNSPECIFIED) {}
//
//        /*!
//            ensures
//                - #type == EUNSPECIFIED
//                - #info == ""
//        !*/
//
//        virtual ~m21error(
//        ) throw() {}
//
//        /*!
//            ensures
//                - does nothing
//        !*/
//
//        const char *what(
//        ) const throw()
//        /*!
//            ensures
//                - if (info.size() != 0) then
//                    - returns info.c_str()
//                - else
//                    - returns type_to_string(type)
//        !*/
//        {
//            if (info.size() > 0)
//                return info.c_str();
//            else
//                return type_to_string();
//        }
//
//        const char *type_to_string(
//        ) const throw()
//        /*!
//            ensures
//                - returns a string that names the contents of the type member.
//        !*/
//        {
//            if (type == EUNSPECIFIED) return "EUNSPECIFIED";
//            else if (type == EBROKEN_ASSERT) return "EBROKEN_ASSERT";
//            else return "undefined error type";
//        }
//
//        const std::string info;  // info about the error
//        const error_type type; // the type of the error
//
//    private:
//        const m21error &operator=(const m21error &);
//    };
//
//// ----------------------------------------------------------------------------------------
//
//    class fatal_error : public m21error {
//        /*!
//            WHAT THIS OBJECT REPRESENTS
//                As the name says, this object represents some kind of fatal error.
//                That is, it represents an unrecoverable error and any program that
//                throws this exception is, by definition, buggy and needs to be fixed.
//
//                Note that a fatal_error exception can only be thrown once.  The second
//                time an application attempts to construct a fatal_error it will be
//                immediately aborted and an error message will be printed to std::cerr.
//                The reason for this is because the first fatal_error was apparently ignored
//                so the second fatal_error is going to make itself impossible to ignore
//                by calling abort.  The lesson here is that you should not try to ignore
//                fatal errors.
//
//                This is also the exception thrown by the MATH21_ASSERT and MATH21_CASSERT macros.
//        !*/
//
//    public:
//        fatal_error(
//                error_type t,
//                const std::string &a
//        ) : m21error(t, a) { check_for_previous_fatal_errors(); }
//
//        /*!
//            ensures
//                - #type == t
//                - #info == a
//        !*/
//
//        fatal_error(
//                error_type t
//        ) : m21error(t) { check_for_previous_fatal_errors(); }
//
//        /*!
//            ensures
//                - #type == t
//                - #info == ""
//        !*/
//
//        fatal_error(
//                const std::string &a
//        ) : m21error(EFATAL, a) { check_for_previous_fatal_errors(); }
//
//        /*!
//            ensures
//                - #type == EFATAL
//                - #info == a
//        !*/
//
//        fatal_error(
//        ) : m21error(EFATAL) { check_for_previous_fatal_errors(); }
//        /*!
//            ensures
//                - #type == EFATAL
//                - #info == ""
//        !*/
//
//    private:
//
//        static inline char *message() {
//            static char buf[2000];
//            buf[1999] = '\0'; // just to be extra safe
//            return buf;
//        }
//
//        static inline void math21_fatal_error_terminate(
//        ) {
//            std::cerr << "\n**************************** FATAL ERROR DETECTED ****************************";
//            std::cerr << message() << std::endl;
//            std::cerr << "******************************************************************************\n"
//                      << std::endl;
//        }
//
//        void check_for_previous_fatal_errors() {
//            // If math21 is being use to create plugins for some other application, like
//            // MATLAB, then don't do these checks since it terminates the over arching
//            // system.  Just let the errors go to the plugin handler and it will deal with
//            // them.
//#if defined(MATLAB_MEX_FILE)
//            return;
//#else
//            static bool is_first_fatal_error = true;
//            if (is_first_fatal_error == false) {
//                std::cerr << "\n\n ************************** FATAL ERROR DETECTED ************************** "
//                          << std::endl;
//                std::cerr << " ************************** FATAL ERROR DETECTED ************************** "
//                          << std::endl;
//                std::cerr << " ************************** FATAL ERROR DETECTED ************************** \n"
//                          << std::endl;
//                std::cerr << "Two fatal errors have been detected, the first was inappropriately ignored. \n"
//                          << "To prevent further fatal errors from being ignored this application will be \n"
//                          << "terminated immediately and you should go fix this buggy program.\n\n"
//                          << "The error message from this fatal error was:\n" << this->what() << "\n\n" << std::endl;
//                using namespace std;
//                assert(false);
//                abort();
//            } else {
//                // copy the message into the fixed message buffer so that it can be recalled by math21_fatal_error_terminate
//                // if needed.
//                char *msg = message();
//                unsigned long i;
//                for (i = 0; i < 2000 - 1 && i < this->info.size(); ++i)
//                    msg[i] = info[i];
//                msg[i] = '\0';
//
//                // set this termination handler so that if the user doesn't catch this math21::fatal_error that is being
//                // thrown then it will eventually be printed to standard error
//                std::set_terminate(&math21_fatal_error_terminate);
//            }
//            is_first_fatal_error = false;
//#endif
//        }
//    };
//
//
//// ----------------------------------------------------------------------------------------
//
//    class impossible_labeling_error : public math21::m21error {
//        /*!
//            WHAT THIS OBJECT REPRESENTS
//                This is the exception thrown by code that trains object detectors (e.g.
//                structural_svm_object_detection_problem) when they detect that the set of
//                truth boxes given to the training algorithm contains some impossible to
//                obtain outputs.
//
//                This kind of problem can happen when the set of image positions scanned by
//                the underlying object detection method doesn't include the truth rectangle
//                as a possible output.  Another possibility is when two truth boxes are very
//                close together and hard coded non-max suppression logic would prevent two
//                boxes in such close proximity from being output.
//        !*/
//    public:
//        impossible_labeling_error(const std::string &msg) : math21::m21error(msg) {};
//    };
//
//// ----------------------------------------------------------------------------------------
//
//}
