#pragma once

/*
 * fatalError:
 *  - message: mensagem descritiva do erro que será exibida antes de encerrar o programa.
 * Encerrra a aplicação imediatamente com falha, útil para abortar execução em condições irreparáveis.
 */
void fatalError(const char *message);

void print_progress(size_t count, size_t max);

char *getFileExtension(char *filename);