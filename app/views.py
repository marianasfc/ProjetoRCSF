from django.shortcuts import redirect, render
from .models import Topico, Conteudo, Conversor
from .forms import TopicoForm, ConteudoForm, ConversorForm
import seaborn as sns
import os
from .funcoes import calculateNyquist, calculateShannon, conversor, erlang_b_numero_canais, erlang_b_tráfego, plot_heatmap_ci, plot_relacao_ci, plot_variacao_atenuacao, plot_variacao_atenuacao_freq, qam, cria_matriz, calcula_matriz_prx, calcula_matriz_espaco_livre, grafico, dadosPlo, espaco_livre, calcular_heatmap_potencia, erlang_b_probabilidade_bloqueio

# Create your views here.
def index_view(request):
    return render(request, 'core/index.html')

def aspetos_view(request):
    return render(request, 'core/aspetos.html')

def topicos_view(request):
    topicos = Topico.objects.all().order_by('ordem')
    context = {
        'topicos':topicos
    }
    return render(request, 'core/topicos.html', context)

def conversor_view(request):
    resultado = "Resultado"
    if request.method == "POST":
        valor = request.POST['valor']
        unidade1 = request.POST['unidade1']
        unidade2 = request.POST['unidade2']
        resultado = conversor(valor, unidade1, unidade2)
    context = {
        'resultado': resultado
    }

    return render(request, 'core/conversor.html', context)

def nyquist_view(request):
    resultado = "Resultado"
    if request.method == "POST":
        b = request.POST['b']
        v = request.POST['v']
        resultado = calculateNyquist(b, v)
    context = {
        'resultado': resultado
    }

    return render(request, 'core/nyquist.html', context)

def shannon_view(request):
    resultado = "Resultado"
    if request.method == "POST":
        b = request.POST['b']
        snr = request.POST['snr']
        resultado = calculateShannon(b, snr)
    context = {
        'resultado': resultado
    }

    return render(request, 'core/shannon.html', context)

def meios_view(request):
    return render(request, 'core/meios.html')

def sinais_view(request):
    return render(request, 'core/sinais.html')

def mpsk_view(request):
    return render(request, 'core/mpsk.html')

def mqam_view(request):
    resultado = "Resultado"
    if request.method == "POST":
        b = request.POST['b']
        db = request.POST['db']
        ber = request.POST['ber']
        resultado = qam(b, db, ber)
    context = {
        'resultado': resultado
    }
    return render(request, 'core/mqam.html', context)

def modulacoes_view(request):
    return render(request, 'core/modulacoes.html')

def aspetos_view(request):
    if os.path.exists('static/core/grafico1.png'):
        os.remove('static/core/grafico1.png')
    if os.path.exists('static/core/grafico2.png'):
        os.remove('static/core/grafico2.png')   
    x_area = 100
    y_area = 100
    d_pixel = 1
    i_centro = 50
    j_centro = 50
    resultado = "Resultado"
    resultado1 = "Resultado1"
    matriz = cria_matriz(x_area, y_area, d_pixel)
    if request.method == "POST":
        ptx= request.POST['ptx']
        f = request.POST['f']
        d = request.POST['d']
        s = request.POST['s']
        prx = calcula_matriz_espaco_livre(d,f,ptx)
        matriz_ptx = calcula_matriz_prx(matriz, prx, f, calcula_matriz_espaco_livre, i_centro, j_centro, d_pixel)
        resultado = grafico(matriz_ptx)
        resultado1 = espaco_livre(ptx, f, d, s)
    context = {
        'resultado': resultado,
        'resultado1': resultado1
    }
    return render(request, 'core/aspetos.html', context)

def cenario_view(request):
    resultado = dadosPlo()
    context = {
        'resultado': resultado
    }
    return render(request, 'core/cenario.html', context)

def diagrama_view(request):
    if os.path.exists('static/core/grafico3.png'):
        os.remove('static/core/grafico3.png')
    resultado = "Resultado"
    if request.method == "POST":
        ptx= request.POST['ptx']
        f = request.POST['f']
        mod = request.POST['mod']
        largura = request.POST['largura']
        altura = request.POST['altura']
        resultado = calcular_heatmap_potencia(ptx, f, mod, largura, altura, tamanho_pixel=10,
                              antena='Omni', azimute_antena=0, altura_antena=20, altura_terminal=1.5)
    context = {
        'resultado': resultado
    }
    return render(request, 'core/diagrama.html', context)

def planeamento_view(request):
    if os.path.exists('static/core/grafico5.png'):
        os.remove('static/core/grafico5.png')
    resultado = "Resultado"
    if request.method == "POST":
        c = request.POST['c']
        f = request.POST['f']
        resultado = plot_variacao_atenuacao_freq(f,c)
    context = {
        'resultado': resultado
    }
    return render(request, 'core/planeamento.html', context)

def plan2_view(request):
    if os.path.exists('static/core/grafico8.png'):
        os.remove('static/core/grafico8.png')
    resultado = "Resultado"
    if request.method == "POST":
        f = request.POST['f']
        resultado = plot_variacao_atenuacao(f)
    context = {
        'resultado': resultado
    }
    return render(request, 'core/plan2.html', context)

def plan3_view(request):
    if os.path.exists('static/core/grafico7.png'):
        os.remove('static/core/grafico7.png')
    resultados1 = plot_relacao_ci()  
    resultado = plot_heatmap_ci()
    context = {
        'resultado': resultado,
        'resultados1': resultados1
    }
    return render(request, 'core/plan3.html', context)

def trafego_view(request):
    return render(request, 'core/trafego.html')

def erlangB_view(request):
    resultado = "Resultado"
    if request.method == "POST":
        t= request.POST['t']
        c = request.POST['c']
        resultado = erlang_b_probabilidade_bloqueio(t, c)
    context = {
        'resultado': resultado,
    }
    return render(request, 'core/erlangB.html', context)

def canais_view(request):
    resultado = "Resultado"
    if request.method == "POST":
        t= request.POST['t']
        pb = request.POST['pb']
        resultado = erlang_b_numero_canais(t, pb)
    context = {
        'resultado': resultado,
    }
    return render(request, 'core/canais.html', context)

def calctraf_view(request):
    resultado = "Resultado"
    if request.method == "POST":
        tc= request.POST['tc']
        dm = request.POST['dm']
        resultado = erlang_b_tráfego(tc, dm)
    context = {
        'resultado': resultado,
    }
    return render(request, 'core/calctraf.html', context)

def edita_topicos_view(request):

    topicos = Topico.objects.all().order_by('ordem')

    context = {
        'topicos':topicos
    }

    return render(request, 'core/edita_topicos.html', context)


def novo_topico_view(request):

    topico_form = TopicoForm(request.POST or None)
    if topico_form.is_valid():
        topico_form.save()
        return redirect('core:edita_topicos')

    context = {'topico_form':topico_form}
    return render(request, 'core/novo_topico.html', context)


def edita_topico_view(request, topico_id):

    topico = Topico.objects.get(id=topico_id)
    topico_form = TopicoForm(request.POST or None, instance=topico)
    if request.method == "POST" and topico_form.is_valid():
        topico_form.save()
        return redirect('core:edita_topicos')

    context = {'topico_form':topico_form, 'topico_id':topico_id}
    return render(request, 'core/edita_topico.html', context)


def apaga_topico_view(request, topico_id):
    topico = Topico.objects.get(id=topico_id)
    topico.delete()
    return redirect('core:edita_topicos')


def novo_conteudo_view(request):

    conteudo_form = ConteudoForm(request.POST or None)
    if conteudo_form.is_valid():
        conteudo_form.save()
        return redirect('core:edita_topicos')

    context = {'conteudo_form':conteudo_form}
    return render(request, 'core/novo_conteudo.html', context)


def edita_conteudo_view(request, conteudo_id):

    conteudo = Conteudo.objects.get(id=conteudo_id)
    conteudo_form = ConteudoForm(request.POST or None, instance=conteudo)
    if request.method == "POST" and conteudo_form.is_valid():
        conteudo_form.save()
        return redirect('core:edita_topicos')

    context = {'conteudo_form':conteudo_form, 'conteudo_id':conteudo_id}
    return render(request, 'core/edita_conteudo.html', context)


def apaga_conteudo_view(request, conteudo_id):
    conteudo = Conteudo.objects.get(id=conteudo_id)
    conteudo.delete()
    return redirect('core:edita_topicos')


