#include "core/helpers.h"
#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
    #include <windows.h>
#else
    #include <unistd.h>
#endif

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

void print_progress(size_t count, size_t max)
{
    const int bar_width = 50;

    float progress = (float)count / max;
    int bar_length = progress * bar_width;

    printf("\rProcessando Amostras: [");
    for (int i = 0; i < bar_length; ++i)
    {
        printf("#");
    }
    for (int i = bar_length; i < bar_width; ++i)
    {
        printf(" ");
    }
    printf("] %.2f%%", progress * 100);

    fflush(stdout);
}

char *getFileExtension(char *filename)
{
    char *dot = strrchr(filename, '.');

    if (!dot || dot == filename || *(dot + 1) == '\0')
        return "";

    return dot + 1;
}