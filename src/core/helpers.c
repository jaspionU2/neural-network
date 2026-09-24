#include "core/helpers.h"
#include <stdio.h>
#include <stdlib.h>

/*
 * fatalError:
 *  - message: mensagem de erro a ser exibida via `perror`
 * Termina o programa chamando `exit(EXIT_FAILURE)`
 */
void fatalError(const char *message)
{
    perror(message);
    exit(EXIT_FAILURE);
}