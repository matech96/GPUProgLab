#pragma once

#include <GL/glew.h>
#include "quad.hpp"

const int vertices_size = 24*30;

class LineStrip
{
protected:
	GLuint vertexArray;

	static int vertecies_pos;
	static GLfloat vertices[vertices_size];
	GLuint vertexBuffer;

	static GLfloat texCoords[12];
	GLuint texCoordBuffer;

public:
	LineStrip();
	~LineStrip();

	void init();
	void addPoint(float x, float y);
	void render();
	void render(int widtgh, int height);
};

//GLfloat LineStrip::vertices[18] = { -0.5f, 0.5f, 0.0f,
//				 0.5f, 0.5f, 0.0f,
//				 0.5f, -0.5f, 0.0f,
//				 -0.5f, -0.5f, 0.0f,
//				 0.0f, 0.0f, 0.0f,
//				 -0.5f, 0.0f, 0.0f };
//-0.5f, -0.5f, 0.0f,
//0.5f, -0.5f, 0.0f,
//0.5f, 0.5f, 0.0f,
//-0.5f, 0.5f, 0.0f
int LineStrip::vertecies_pos = 0;
GLfloat LineStrip::vertices[vertices_size] = { };

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

	glGenBuffers(1, &texCoordBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, texCoordBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 12, texCoords, GL_STATIC_DRAW);
	glEnableVertexAttribArray(2);
	glVertexAttribPointer((GLuint)2, 2, GL_FLOAT, GL_FALSE, 0, 0);

	glBindVertexArray(0);
}

inline void LineStrip::addPoint(float x, float y)
{
	if (vertecies_pos >= 4*3){
		int vp = vertecies_pos;
		for (int i = 3*3; i > 0; i--)
		{
			vertices[vertecies_pos++] = vertices[vp - i];
		}
	}
	vertices[vertecies_pos++] = x;
	vertices[vertecies_pos++] = y;
	vertices[vertecies_pos++] = 0.0f;
}


void LineStrip::render() {

	glGenVertexArrays(1, &vertexArray);
	glBindVertexArray(vertexArray);

	glGenBuffers(1, &vertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * vertices_size, vertices, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer((GLuint)0, 3, GL_FLOAT, GL_FALSE, 0, 0);


	glBindVertexArray(vertexArray);
	glDrawArrays(GL_LINES_ADJACENCY, 0, vertecies_pos / 3);
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

