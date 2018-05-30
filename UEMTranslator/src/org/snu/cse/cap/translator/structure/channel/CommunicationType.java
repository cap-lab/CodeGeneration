package org.snu.cse.cap.translator.structure.channel;

public enum CommunicationType {
	SHARED_MEMORY,
	TCP,
	TCP_CLIENT,
	TCP_SERVER,
	CPU_GPU,
	GPU_CPU,
	GPU_GPU,
	GPU_GPU_DIFFERENT,
}