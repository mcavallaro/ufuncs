#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"

static PyMethodDef nmfMethods[] = {
        {NULL, NULL, 0, NULL}
};


static void double_nmf(char **args, npy_intp *dimensions, npy_intp* steps, void* data) {
    npy_intp l = dimensions[0];

    printf("%d\n", (int)l); /*this is the dimension of the output vector*/

    printf("input1: %d\n",*(int*)(args[0] + steps[0]));// input
    printf("input2: %f\n",*(double*)args[1]); // input

    *(double*)args[2] = 1; //output
    *(double*)(args[2] + steps[2]) = 81.9;
    *(double*)(args[2] + 2*steps[2]) = 19;

    // just print input matrix
    // printf("Input matrix:\n");
    // for(i=0;i<n;i++){
    //     for(j=0;j<m;j++){
    //         printf("%.1f ",*(double*)(args[0]+8*(i*m+j)));
    //     }
    // printf("\n");
    // }
    return;

}

static PyUFuncGenericFunction nmf_functions[] = { double_nmf };
static void * nmf_data[] = {(void *)NULL};
static char nmf_types[] = {PyArray_INT64, PyArray_DOUBLE, PyArray_DOUBLE}; /*args[0], args[1], args[2]*/
//char *nmf_signature = "(n),()->(n)";

PyMODINIT_FUNC initnmf(void) {
    PyObject *m, *d, *version, *nmf;

    m = Py_InitModule("nmf", nmfMethods);
    if (m == NULL) {
        return;
    }

    import_array();
    import_umath();
    d = PyModule_GetDict(m);
    version = PyString_FromString("0.1");
    PyDict_SetItemString(d, "__version__", version);
    Py_DECREF(version);

    nmf = PyUFunc_FromFuncAndData(nmf_functions, nmf_data, nmf_types, 1,
                                  2, /*number of input*/
                                  1, /*number of output*/
                                  PyUFunc_None, "nmf", "", 0);

    PyDict_SetItemString(d, "nmf", nmf);
    Py_DECREF(nmf);
}
