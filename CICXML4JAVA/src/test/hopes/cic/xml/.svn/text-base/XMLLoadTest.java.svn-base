package hopes.cic.xml;

import java.io.File;
import java.math.BigInteger;

public class XMLLoadTest {
	final static String ALGORITHM_FILE = ".\\samples\\toy\\CIC_algo.xml";
	final static String ARCHITECTURE_FILE = ".\\samples\\toy\\CIC_arc.xml";
	final static String MAP_FILE = ".\\samples\\toy\\CIC_map.xml";
	final static String PROFILE_FILE = ".\\samples\\toy\\CIC_pro.xml";
	
//	final static String ALGORITHM_FILE = ".\\samples\\port_test\\PTHREAD_CIC.algo";
//	final static String ARCHITECTURE_FILE = ".\\samples\\port_test\\PTHREAD_CIC.arch";
//	final static String MAP_FILE = ".\\samples\\port_test\\PTHREAD_CIC.map";
	
	public static void main(String[] args) throws Exception {
		{
			CICAlgorithmTypeLoader loader = new CICAlgorithmTypeLoader();

			CICAlgorithmType algorithm = loader.loadResource(ALGORITHM_FILE);
			System.out.println("algorithm : " + algorithm);
			
			// TODO maniplulate algorithm data
			algorithm.getChannels().getChannel().get(0).setSize(new BigInteger("1000000"));
			
			loader.storeResource(algorithm, new File(ALGORITHM_FILE + ".copy"));
		}
		{
			CICArchitectureTypeLoader loader = new CICArchitectureTypeLoader();

			CICArchitectureType architecture = loader.loadResource(ARCHITECTURE_FILE);
			System.out.println("architecture : " + architecture);
			
			// TODO maniplulate architecture data
			architecture.setTarget("AAAAA");

			loader.storeResource(architecture, new File(ARCHITECTURE_FILE + ".copy"));
		}
		{
			CICMappingTypeLoader loader = new CICMappingTypeLoader();

			CICMappingType mapping = loader.loadResource(MAP_FILE);
			System.out.println("mapping : " + mapping);
			
			// TODO maniplulate mapping data
			mapping.getTask().get(0).setName("nameBBBB");
			
			loader.storeResource(mapping, new File(MAP_FILE + ".copy"));
		}
		{
			CICProfileTypeLoader loader = new CICProfileTypeLoader();

			CICProfileType profile = loader.loadResource(PROFILE_FILE);
			System.out.println("profile : " + profile);
			
			// TODO maniplulate profile data
			profile.getTask().get(0).setName("nameCCCC");
			loader.storeResource(profile, new File(PROFILE_FILE + ".copy"));
		}

	}
}
