package org.snu.cse.cap.translator.structure.device.connection;

public class SSLTCPConnection extends IPConnection {
	private int channelAccessNum;
	private SSLKeyInfo sslKeyInfo;
	private int sslKeyInfoIndex;

	public SSLTCPConnection(String name, String role, String IP, int port, String caPublicKey, String publicKey,
			String privateKey) {
		super(name, role, IP, port, ProtocolType.SECURE_TCP);
		this.channelAccessNum = 0;
		this.sslKeyInfo = new SSLKeyInfo(caPublicKey, publicKey, privateKey);
	}

	public int getChannelAccessNum() {
		return channelAccessNum;
	}

	public void incrementChannelAccessNum() {
		this.channelAccessNum = this.channelAccessNum + 1;
	}

	public SSLKeyInfo getSSLKeyInfo() {
		return sslKeyInfo;
	}

	public int getKeyInfoIndex() {
		return sslKeyInfoIndex;
	}

	public void setKeyInfoIndex(int sslKeyInfoIndex) {
		this.sslKeyInfoIndex = sslKeyInfoIndex;
	}
}
