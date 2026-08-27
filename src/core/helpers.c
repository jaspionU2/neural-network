#include "core/helpers.h"
#include <stdio.h>
#include <stdlib.h>

void fatalError(const char *message)
{
    perror(message);
    exit(EXIT_FAILURE);
}