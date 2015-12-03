package hopes.cic.xml;
import java.net.URL;


public class ClassLoaderTest {
	
	public static void main(String[] args) {
		ClassLoaderTest test = new ClassLoaderTest();
		
		URL url = test.getClass().getClassLoader().getResource(".project");
		System.out.println(url);
		
		URL url2 = test.getClass().getClassLoader().getResource("xsd/CICCommon.xsd");
		System.out.println(url2);	
		
	}
}
