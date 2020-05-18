// Skeleton.cpp : Defines the entry point for the console application.
//

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <glm/gtc/matrix_transform.hpp>
#include <random>
#include <algorithm>
#include <iterator>

#include <cstdio>
#include <algorithm>

#include "shader.hpp"
#include "texture.hpp"
#include "DebugOpenGL.hpp"

const unsigned int windowWidth = 600;
const unsigned int windowHeight = 600;

// Grid Resolution (resX=resY)
const unsigned int N = 64;
// Workgroup Size
const unsigned int Nwg = 32;
// Number of vertices (resX=resY)
const unsigned int vNum = N*N;

const float center[3] = {0.0, -0.5, 0.0};
const float r = 0.5;
const float d = 1.0 / (N-1);

// Vec4 like structure
struct xyzw
{
	float x, y, z, w;
};

//compute shader
Shader vertexGravityShader;
Shader vertexCollisionShader;
Shader vertexDistanceShader;
Shader vertexBendingShader;
Shader vertexFinalUpdateShader;
//standard pipeline
Shader renderShader;

// Position buffer
GLuint positionBuffer; //x
GLuint positionBufferTmp; //p
GLuint velocityBuffer; //v

// Vertex array object
GLuint vao;

void onInitialization()
{
	glewExperimental = true;
	if (glewInit() != GLEW_OK)
	{
		printf("Cannot initialize GLEW\n");
		exit(-1);
	}
	glGetError();

	DebugOpenGL::init();
	DebugOpenGL::enableLowSeverityMessages(false);

	glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

	vertexGravityShader.loadShader(GL_COMPUTE_SHADER, "../shaders/gravity.comp");
	vertexGravityShader.compile();

	vertexCollisionShader.loadShader(GL_COMPUTE_SHADER, "../shaders/collision.comp");
	vertexCollisionShader.compile();

	vertexDistanceShader.loadShader(GL_COMPUTE_SHADER, "../shaders/distance.comp");
	vertexDistanceShader.compile();

	vertexBendingShader.loadShader(GL_COMPUTE_SHADER, "../shaders/bending.comp");
	vertexBendingShader.compile();

	vertexFinalUpdateShader.loadShader(GL_COMPUTE_SHADER, "../shaders/finalUpdate.comp");
	vertexFinalUpdateShader.compile();

	renderShader.loadShader(GL_VERTEX_SHADER, "../shaders/render.vert");
	renderShader.loadShader(GL_FRAGMENT_SHADER, "../shaders/render.frag");
	renderShader.compile();

	// Initialize the particle position buffer
	glGenBuffers(1, &positionBuffer);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, positionBuffer);
	glBufferData(GL_SHADER_STORAGE_BUFFER, vNum * sizeof(xyzw), NULL, GL_STATIC_DRAW);
	xyzw* pos = (xyzw*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, vNum * sizeof(xyzw), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
	for (unsigned int i = 0; i < N; ++i)
		for (unsigned int j = 0; j < N; ++j)
	{
		pos[j*N+i].x = (double)i / ((double)N - 1.0) - 0.5;
		pos[j*N+i].y = 0;
		pos[j*N+i].z = (double)j / ((double)N - 1.0) - 0.5;
		pos[j*N+i].w = 1.0f;
	}
	glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

	glGenBuffers(1, &positionBufferTmp);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, positionBufferTmp);
	glBufferData(GL_SHADER_STORAGE_BUFFER, vNum * sizeof(xyzw), NULL, GL_STATIC_DRAW);
	xyzw* posTmp = (xyzw*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, vNum * sizeof(xyzw), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
	glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

	glGenBuffers(1, &velocityBuffer);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, velocityBuffer);
	glBufferData(GL_SHADER_STORAGE_BUFFER, vNum * sizeof(xyzw), NULL, GL_STATIC_DRAW);
	xyzw* vel = (xyzw*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, vNum * sizeof(xyzw), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
	for (unsigned int i = 0; i < N*N; ++i)
	{
		vel[i].x = 0;
		vel[i].y = 0;
		vel[i].z = 0;
		vel[i].w = 0;
	}
	glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
			
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	// Initialize the vertex array object with the position and velocity buffers
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, positionBuffer);
	glVertexAttribPointer((GLuint)0, 4, GL_FLOAT, GL_FALSE, sizeof(xyzw), (GLvoid*)0);

	glBindVertexArray(0);

	// Set point primitive size
	glPointSize(8.0f);
}

void onDisplay()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glm::mat4 view = glm::lookAt(glm::vec3(0, 0.5, 2.5), glm::vec3(0, -0.5, 0.5), glm::vec3(0, 1, 0));
	glm::mat4 proj = glm::perspective(glm::radians(45.0f), (float)windowWidth/ (float)windowHeight, 0.1f, 10.0f);
		
	const float dt = 0.0001f;
	//TODO call External force shader
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, positionBufferTmp);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, positionBuffer);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, velocityBuffer);
	vertexGravityShader.enable();
	vertexGravityShader.setUniform1f("dt", dt);
	vertexGravityShader.setUniform1i("N", N);
	glDispatchCompute(N / Nwg, N / Nwg, 1);
	
	float collWeight = 1.0, sideDistWeight = 0.1, diagDistWeight = 0.05, sideBendWeight = 0.001, diagBendWeight = 0.0005;

	const int NITER = 50;
	for (int i = 0; i < NITER; ++i)
	{
		//TODO call constraint shaders
		
		vertexCollisionShader.enable();
		vertexCollisionShader.setUniform1f("collWeight", collWeight);
		vertexCollisionShader.setUniform1f("r", r);
		vertexCollisionShader.setUniform3f("center", center[0], center[1], center[2]);
		vertexCollisionShader.setUniform1i("N", N);
		glDispatchCompute(N / Nwg, N / Nwg, 1);
		
		vertexDistanceShader.enable();
		vertexDistanceShader.setUniform1f("sideDistWeight", sideDistWeight);
		vertexDistanceShader.setUniform1f("diagDistWeight", diagDistWeight);
		vertexDistanceShader.setUniform1f("d", d);
		vertexDistanceShader.setUniform1i("N", N);
		glDispatchCompute(N / Nwg, N / Nwg, 1);

		/*vertexBendingShader.enable();
		vertexBendingShader.setUniform1i("N", N);
		vertexBendingShader.setUniform1f("sideBendWeight", sideBendWeight);
		vertexBendingShader.setUniform1f("diagBendWeight", diagBendWeight);
		glDispatchCompute(N / Nwg, N / Nwg, 1);*/


	}
	
	//TODO call final update Shader
	vertexFinalUpdateShader.enable();
	vertexFinalUpdateShader.setUniform1f("dt", dt);
	vertexFinalUpdateShader.setUniform1i("N", N);
	glDispatchCompute(N / Nwg, N / Nwg, 1);
	
	// Render the particles
	renderShader.enable();
	renderShader.setUniformMat4("viewproj", proj*view);
	glBindVertexArray(vao);
	glEnable(GL_BLEND); 
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glDrawArrays(GL_POINTS, 0, N*N);
	glBindVertexArray(0);
	renderShader.disable();

	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
	switch (key)
	{
	case 27:
		glutExit();
		break;
	}
}

void onIdle()
{
	glutPostRedisplay();
}

int main(int argc, char* argv[])
{
	glutInit(&argc, argv);

	glutInitContextVersion(4, 3);
	glutInitWindowSize(windowWidth, windowHeight);
	glutInitWindowPosition(100, 100);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
	glutCreateWindow(argv[0]);
	glewExperimental = true;
	glewInit();

	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	GLint major, minor;
	glGetIntegerv(GL_MAJOR_VERSION, &major);
	glGetIntegerv(GL_MINOR_VERSION, &minor);
	printf("GL Version (integer) : %d.%d\n", major, minor);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	onInitialization();
	glutDisplayFunc(onDisplay);
	glutKeyboardFunc(onKeyboard);
	glutIdleFunc(onIdle);
	glutMainLoop();

	return 0;
}
