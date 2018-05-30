package org.snu.cse.cap.translator.structure.channel;

public enum CommunicationType {
	SHARED_MEMORY,
	TCP_CLIENT_READER,
	TCP_SERVER_WRITER,
	TCP_CLIENT_WRITER,
	TCP_SERVER_READER,	
	CPU_GPU,
	GPU_CPU,
	GPU_GPU,
	GPU_GPU_DIFFERENT,
}