
#include <stdio.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_randist.h>
#include <math.h>
#include <float.h>
#include <Python.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"

struct my_f_params
{
  int n;
  double Aoff;
  double Aon;
  double Kon;
  double Koff;
};

//As for uPoissonBeta_pmf_types, needs to work on a 64bit system. Otherwise modify PyArray_INT64

/*First definition of the PoissonBeta with four parameters */
double PoissonBeta_pmf(int k, double Aoff, double Aon, double Kon, double Koff){
  long double sum;
  int i=0;
  long double factor1, factor2;

  factor1 = (k - i)*logl(Aoff) + i*logl(Aon - Aoff) + gsl_sf_lngamma(Kon + i) -
    gsl_sf_lnfact(k - i) - gsl_sf_lnfact(i)  -  gsl_sf_lngamma(Kon +
     Koff + i);

  factor2 = gsl_sf_hyperg_1F1(Kon + i, Kon + Koff + i, Aoff - Aon);

  sum =  expl(factor1) * factor2;
  for (i=1; i<=k; i++){
    factor1 = logl( powl(Aoff, k-i) * powl((Aon - Aoff), i) ) + gsl_sf_lngamma(Kon + i) 
       - gsl_sf_lnfact(k - i) - gsl_sf_lnfact(i) - gsl_sf_lngamma(Kon + Koff + i);
    factor2 = gsl_sf_hyperg_1F1(Kon + i, Kon + Koff + i, Aoff - Aon);
    sum = sum + expl(factor1)* factor2;
  }

  return (double)expl( -Aoff + gsl_sf_lngamma(Kon + Koff) - gsl_sf_lngamma(Kon)) * sum;
}

static void uPoissonBeta_pmf(char **args, npy_intp *dimensions, npy_intp * steps, void *data){
  npy_intp l = dimensions[0];
  npy_intp i;
  for (i=0; i<l; i++){
    *(double*)(args[5] + i*steps[5]) = PoissonBeta_pmf(*(int*)(args[0] + i*steps[0]), *(double*)args[1], *(double*)args[2], *(double*)args[3], *(double*)args[4]);
  }
}

/*Definition of the PoissonBeta with three parameters */
double PoissonBeta1_pmf(int k, double Aon, double Kon, double Koff){
  long double factor1, factor2;

  factor1 = logl(powl(Aon, k)) + gsl_sf_lngamma(Kon + k) 
      - gsl_sf_lnfact(k) - gsl_sf_lngamma(Kon + Koff + k);
  factor2 = gsl_sf_hyperg_1F1(Kon + k, Kon + Koff + k, - Aon);

  return (double)expl( gsl_sf_lngamma(Kon + Koff) - gsl_sf_lngamma(Kon)) * expl(factor1) * factor2;
}

static void uPoissonBeta1_pmf(char **args, npy_intp *dimensions, npy_intp * steps, void *data){
  npy_intp l = dimensions[0];
  npy_intp i;
  for (i=0; i<l; i++){
    *(double*)(args[4] + i*steps[4]) = PoissonBeta1_pmf(*(int*)(args[0] + i*steps[0]), *(double*)args[1], *(double*)args[2], *(double*)args[3]);
  }
}


/*Alternative definition of the PoissonBeta with four parameters */
double PoissonBeta2_integrand(double x, void *params){
  struct my_f_params * p = (struct my_f_params *)params;
  double uno = gsl_ran_poisson_pdf(p->n, (p->Aon - p->Aoff)*x + p->Aoff);
  double due = gsl_ran_beta_pdf(x, p->Kon, p->Koff);
  return uno * due;
}

double PoissonBeta2_pmf(int k, double Aoff, double Aon, double Kon, double Koff){
  gsl_set_error_handler_off();
  int status;
  gsl_integration_workspace * w = gsl_integration_workspace_alloc(1000);
  gsl_function F;
  double result, error;
  F.function = &PoissonBeta2_integrand;
  struct my_f_params params = {k, Aoff, Aon, Kon, Koff};
  F.params = &params;
  status = gsl_integration_qags(&F, 0, 1, 0, 1e-5, 1000,  w, &result, &error);
  if (status){
    printf("shit\n");
  }
  gsl_integration_workspace_free(w);
  return result;
}

static void uPoissonBeta2_pmf(char **args, npy_intp *dimensions, npy_intp * steps, void *data){
  npy_intp l = dimensions[0];
  npy_intp i;
  for (i=0; i<l; i++){
    *(double*)(args[5] + i*steps[5]) = PoissonBeta2_pmf(*(int*)(args[0] + i*steps[0]), *(double*)args[1], *(double*)args[2], *(double*)args[3], *(double*)args[4]);
  }
}


double ShiftedLogNormal_pdf(double x, double shift, double mu, double sigma2){
  if (x <= -shift){
    return 0;
  }
  else{
    return exp(- pow((log(x + shift) - mu),2)/(2*sigma2)) / ((x + shift) * sqrt(sigma2 * 2 * M_PI));
  }
}

static void uShiftedLogNormal_pdf (char **args, npy_intp *dimensions, npy_intp * steps, void *data){
  npy_intp l = dimensions[0];
  npy_intp i;
  for (i=0; i<l; i++){
    *(double*)(args[4] + i*steps[4]) = ShiftedLogNormal_pdf(*(double*)(args[0] + i*steps[0]), *(double*)args[1], *(double*)args[2], *(double*)args[3]);
  }
}

static void uMarginal_pdf(char **args, npy_intp *dimensions, npy_intp * steps, void *data){
  npy_intp s;
  npy_intp n = dimensions[0];

  double kappa = *(double*)args[1];
  double a0 = *(double*)args[2];
  double a1 = *(double*)args[3];
  double Kon = *(double*)args[4];
  double Koff = *(double*)args[5];
  double shift = *(double*)args[6];
  double mu = *(double*)args[7];
  double sigma2 = *(double*)args[8];

  if ((a0 > a1) || (a0 < 0) || (a1 <= 0) || (Koff <= 0) || (Kon <= 0)){
    for(s=0; s<n; s++){ 
      *(double*)(args[9] + s*steps[9]) = 0;
    }
  }
  else{
    int i;
    double tmp1, tmp2;
    double sum;
    double y;
    double I;
    for(s=0; s<n; s++){
      y = *(double*)(args[0] + s*steps[0]);
      I = (shift + y) / kappa;
      sum = 0;
      for (i=0; i<=I; i++){
        tmp2 = PoissonBeta_pmf(i, a0, a1, Kon, Koff);
        tmp1 = ShiftedLogNormal_pdf(y - i*kappa, shift, mu, sigma2);
        sum = sum + tmp1*tmp2;
      }
      *(double*)(args[9] + s*steps[9]) = sum;
    }
  }
}

static PyUFuncGenericFunction uPoissonBeta_pmf_func[] = {uPoissonBeta_pmf};
static void * uPoissonBeta_pmf_data[] = {(void*)NULL};
static char uPoissonBeta_pmf_types[] = {PyArray_INT64, PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE};
//char *nmf_signature = "(n),()->(n)";

static PyUFuncGenericFunction uPoissonBeta1_pmf_func[] = {uPoissonBeta1_pmf};
static void * uPoissonBeta1_pmf_data[] = {(void*)NULL};
static char uPoissonBeta1_pmf_types[] = {PyArray_INT64, PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE};
//char *nmf_signature = "(n),()->(n)";

static PyUFuncGenericFunction uPoissonBeta2_pmf_func[] = {uPoissonBeta2_pmf};
static void * uPoissonBeta2_pmf_data[] = {(void*)NULL};
static char uPoissonBeta2_pmf_types[] = {PyArray_INT64, PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE};
//char *nmf_signature = "(n),()->(n)";

static PyUFuncGenericFunction uShiftedLogNormal_pdf_func[] = {uShiftedLogNormal_pdf};
static void * uShiftedLogNormal_pdf_data[] = {(void *)NULL};
static char uShiftedLogNormal_pdf_types[] = {PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE};
//char *nmf_signature = "(n),()->(n)";

static PyUFuncGenericFunction uMarginal_pdf_func[] = {uMarginal_pdf};
static void * uMarginal_pdf_data[] = {(void *)NULL};
static char uMarginal_pdf_types[] = {PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE};
//char *nmf_signature = "(n),()->(n)";

// define functions in module
static PyMethodDef myMethods[] =
{
  {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initUNpPoissonBetaUtils(void) {
  PyObject *m, *d, *version;
  PyObject *uPoissonBeta_pmf, *uPoissonBeta1_pmf, *uPoissonBeta2_pmf, *uShiftedLogNormal_pdf, *uMarginal_pdf;

  m = Py_InitModule("UNpPoissonBetaUtils", myMethods); //in python3 this doesn't work
  if (m == NULL) {
      return;
  }

  import_array();
  import_umath();

  d = PyModule_GetDict(m);
  version = PyString_FromString("0.1");
  PyDict_SetItemString(d, "__version__", version);
  Py_DECREF(version);

  uPoissonBeta_pmf = PyUFunc_FromFuncAndData(uPoissonBeta_pmf_func, uPoissonBeta_pmf_data, uPoissonBeta_pmf_types, 2,
                                  5, /*number of input*/
                                  1, /*number of output*/
                                  PyUFunc_None, "PoissonBeta_pmf", "", 0);

  uPoissonBeta1_pmf = PyUFunc_FromFuncAndData(uPoissonBeta1_pmf_func, uPoissonBeta1_pmf_data, uPoissonBeta1_pmf_types, 2,
                                  4, /*number of input*/
                                  1, /*number of output*/
                                  PyUFunc_None, "PoissonBeta1_pmf", "", 0);

  uPoissonBeta2_pmf = PyUFunc_FromFuncAndData(uPoissonBeta2_pmf_func, uPoissonBeta2_pmf_data, uPoissonBeta2_pmf_types, 2,
                                  5, /*number of input*/
                                  1, /*number of output*/
                                  PyUFunc_None, "PoissonBeta2_pmf", "", 0);

  uShiftedLogNormal_pdf = PyUFunc_FromFuncAndData(uShiftedLogNormal_pdf_func, uShiftedLogNormal_pdf_data, uShiftedLogNormal_pdf_types, 1,
                                  4, /*number of input*/
                                  1, /*number of output*/
                                  PyUFunc_None, "ShiftedLogNormal_pdf", "", 0);

  uMarginal_pdf = PyUFunc_FromFuncAndData(uMarginal_pdf_func, uMarginal_pdf_data, uMarginal_pdf_types, 1,
                                  9,  /*number of input*/
                                  1, /*number of output*/
                                  PyUFunc_None, "Marginal_pdf", "", 0);

  PyDict_SetItemString(d, "uPoissonBeta_pmf", uPoissonBeta_pmf);
  PyDict_SetItemString(d, "uPoissonBeta1_pmf", uPoissonBeta1_pmf);
  PyDict_SetItemString(d, "uPoissonBeta2_pmf", uPoissonBeta2_pmf);
  PyDict_SetItemString(d, "uShiftedLogNormal_pdf", uShiftedLogNormal_pdf);
  PyDict_SetItemString(d, "uMarginal_pdf", uMarginal_pdf);

  Py_DECREF(uPoissonBeta_pmf);
  Py_DECREF(uPoissonBeta1_pmf);
  Py_DECREF(uPoissonBeta2_pmf);
  Py_DECREF(uShiftedLogNormal_pdf);
  Py_DECREF(uMarginal_pdf);
}

/*static PyObject* NumPyPoissonBeta_pmf(PyObject* self, PyObject* args){
  int k;
  double a0, a1, Kon, Koff;
  double value;
  // Parse the input tuple 
  if (!PyArg_ParseTuple(args, "idddd", &k, &a0, &a1, &Kon, &Koff))
    return NULL;
  value = PoissonBeta_pmf(k, a0, a1, Kon, Koff);
  return Py_BuildValue("f", value);
}

static PyObject* NumPyShiftedLogNormal_pdf(PyObject* self, PyObject* args){
  double x, shift, mu, sigma2;
  double value;
  if (!PyArg_ParseTuple(args, "dddd", &x, &shift, &mu, &sigma2))
    return NULL;
  value = ShiftedLogNormal_pdf(x, shift, mu, sigma2);
  return Py_BuildValue("f", value);
}
*/
/*static PyObject* NumPyMarginal_pdf(PyObject* self, PyObject* args){
  //This function makes use of Numpy C-API
  double a0, a1, Kon, Koff;
  double kappa, shift, mu, sigma2;
  int N;
  PyArrayObject *y_input;
  PyArrayObject *output;

  if (!PyArg_ParseTuple(args, "O!dddddddd", &PyArray_Type, &y_input, &kappa, &a0, &a1, &Kon, &Koff, &shift, &mu, &sigma2))
    return NULL;

  output = (PyArrayObject*)PyArray_FromDims(1, (int*)y_input->dimensions, NPY_DOUBLE);

  N = y_input->dimensions[0];

//  printf("%f %f\n",   ((double*)PyArray_GetPrt(y_input, 0)[0],   *(double*)PyArray_GETPTR1(y_input, 1) );

  if (!Marginal_pdf( (double*)output->data, (double*)y_input->data, N, kappa, a0, a1, Kon, Koff, shift, mu, sigma2)){
    return NULL;
  }

  return PyArray_Return(output);
}

//  define functions in module
static PyMethodDef myMethods[] =
{
  {"NumPyPoissonBeta_pmf", NumPyPoissonBeta_pmf, METH_VARARGS, "evaluate the PoissonBeta PDF"},
  {"NumPyShiftedLogNormal_pdf", NumPyShiftedLogNormal_pdf, METH_VARARGS, "evaluate the ShiftedLogNormal PDF"},
  {"NumPyMarginal_pdf", NumPyMarginal_pdf, METH_VARARGS, "evaluate the marginal PDF"},
  {NULL, NULL, 0, NULL}
};

// module initialization
PyMODINIT_FUNC initNpPoissonBetaUtils(void){
  PyObject *m, *ufMarginal_pdf, *d; 

  m = Py_InitModule("UNpPoissonBetaUtils", myMethods);
  if (m==NULL){
    return;
  }
  import_array();
  import_umath();

  ufMarginal_pdf = PyUFunc_FromFuncAndData(funcs, data, types, 1, 9, 1, PyUFunc_None, "");
  d = PyModule_GetDict(m);

  PyDict_SetItemString(d, "ufMarginal_pdf", ufMarginal_pdf);
  Py_DECREF(logit);
}
*/



/*int main(void){
  double a0 = 1;
  double a1 = 1.1;
  double Kon = 1.1;
  double Koff = 9;
  int i;
  for (i=0; i<10; i++){
    printf("%d %e\n", i+1, PoissonBeta_pmf(i, 0.1, 1.1, 110, 190));
  }
  return 0;
}*/
