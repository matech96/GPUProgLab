#pragma once

#include <GL/glew.h>
#include "quad.hpp"

class LineStrip
{
protected:
	GLuint vertexArray;

	static GLfloat vertices[18];
	GLuint vertexBuffer;

	static GLfloat texCoords[12];
	GLuint texCoordBuffer;

public:
	LineStrip();
	~LineStrip();

	void init();
	void render();
	void render(int widtgh, int height);
};

GLfloat LineStrip::vertices[18] = { -0.5f, 0.5f, 0.0f,
				 0.5f, 0.5f, 0.0f,
				 0.5f, -0.5f, 0.0f,
				 -0.5f, -0.5f, 0.0f,
				 0.0f, 0.0f, 0.0f,
				 -0.5f, 0.0f, 0.0f };

GLfloat LineStrip::texCoords[12] = { 0.0f, 1.0f,
				   1.0f, 1.0f,
				   0.0f, 0.0f,
				   0.0f, 0.0f,
				   1.0f, 1.0f,
				   1.0f, 0.0f };

LineStrip::LineStrip()
{
}

LineStrip::~LineStrip()
{
	glDeleteBuffers(1, &vertexBuffer);
	glDeleteVertexArrays(1, &vertexArray);
	glDeleteVertexArrays(1, &texCoordBuffer);
}

void LineStrip::init()
{
	glGenVertexArrays(1, &vertexArray);
	glBindVertexArray(vertexArray);

	glGenBuffers(1, &vertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 18, vertices, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer((GLuint)0, 3, GL_FLOAT, GL_FALSE, 0, 0);

	glGenBuffers(1, &texCoordBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, texCoordBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 12, texCoords, GL_STATIC_DRAW);
	glEnableVertexAttribArray(2);
	glVertexAttribPointer((GLuint)2, 2, GL_FLOAT, GL_FALSE, 0, 0);

	glBindVertexArray(0);
}

void LineStrip::render() {
	glBindVertexArray(vertexArray);
	glDrawArrays(GL_LINE_STRIP, 0, 6);
	glBindVertexArray(0);
}

void LineStrip::render(int width, int height)
{
	int vp[4];
	glGetIntegerv(GL_VIEWPORT, vp);
	glViewport(0, 0, width, height);
	render();
	glViewport(vp[0], vp[1], vp[2], vp[3]);
}

